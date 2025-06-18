import time
import requests
import json
import random
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import asyncio
# import websockets # <-- WEBSOCKETS KALDIRILDI
# import ssl # <-- SSL KALDIRILDI

from aiohttp import web # <-- aiohttp eklendi

# components dosyasından sınıfları içe aktar
from electric_bus_components import ElectricMotor, Battery, Driver, Environment, FaultManager, RouteManager, apply_noise

# --- JSON Encoder for NumPy types ---
class NumpyEncoder(json.JSONEncoder):
    """
    Numpy tiplerini JSON serileştirilebilir Python tiplerine dönüştürmek için özel kodlayıcı.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray): # Array'leri de listeye çevirir (eğer varsa)
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
# --- END Numpy Encoder ---

# .env dosyasından ortam değişkenlerini yükle
load_dotenv()

# Next.js API endpoint'i (MongoDB'ye veri göndermek için hala gerekli)
NEXTJS_API_URL = os.getenv("NEXTJS_API_URL", "http://localhost:3000/api/can-data")

# Python HTTP Komut Sunucusu Ayarları
COMMAND_SERVER_PORT = 8766 # Yeni bir port, WebSocket portundan farklı
COMMAND_SERVER_HOST = "0.0.0.0"

BUS_ID = "EV_TR001" # Otobüs ID'si

print(f"Elektrikli Otobüs Simülatörü başlatılıyor. Veriler {NEXTJS_API_URL} adresine gönderilecek.")
print(f"HTTP Komut Sunucusu {COMMAND_SERVER_HOST}:{COMMAND_SERVER_PORT} üzerinde çalışacak.")

# --- SİMÜLASYON PARAMETRELERİ ---
SIMULATION_INTERVAL_MS = 1
MAX_SIMULATION_DURATION_SECONDS = 3600 * 24 * 5

# --- OTOBÜS FİZİKSEL SABİTLERİ ---
MASS_KG = 18000
DRAG_COEFFICIENT = 0.6
FRONTAL_AREA_SQM = 7.0
ROLLING_RESISTANCE_COEFF = 0.01
GRAVITY = 9.81

# --- SİMÜLASYON NESNELERİNİ BAŞLAT ---
motor = ElectricMotor()
battery = Battery()
driver = Driver(profile="normal", target_speed_kph=0)
environment = Environment(initial_temp=25)
fault_manager = FaultManager()

# --- ÖRNEK ROTA VERİSİ ---
EXAMPLE_ROUTE_DATA = [
    {"distance_km": 0.1, "slope_degrees": 0, "speed_limit_kph": 0, "traffic_density": "low", "action": "start"},
    {"distance_km": 5, "slope_degrees": 0, "speed_limit_kph": 50, "traffic_density": "medium", "action": "drive_city"},
    {"distance_km": 20, "slope_degrees": 2, "speed_limit_kph": 80, "traffic_density": "low", "action": "drive_highway_uphill"},
    {"distance_km": 30, "slope_degrees": -1, "speed_kph": 90, "traffic_density": "low", "action": "drive_highway_downhill"},
    {"distance_km": 0.5, "slope_degrees": 0, "speed_limit_kph": 0, "traffic_density": "high", "action": "stop_traffic_light"},
    {"distance_km": 150, "slope_degrees": 0.5, "speed_limit_kph": 90, "traffic_density": "low", "action": "drive_long_haul"},
    {"distance_km": 1.0, "slope_degrees": 0, "speed_limit_kph": 0, "traffic_density": "low", "action": "charge_station"},
    {"distance_km": 100, "slope_degrees": -0.5, "speed_limit_kph": 85, "traffic_density": "low", "action": "drive_after_charge"},
    {"distance_km": 0.1, "slope_degrees": 0, "speed_limit_kph": 0, "traffic_density": "low", "action": "end_route"}
]
route_manager = RouteManager(EXAMPLE_ROUTE_DATA)

# --- ARIZA SENARYOLARINI TANIMLA (ÖRNEK) ---
fault_manager.add_fault_trigger("battery_overheat", trigger_time_seconds=3600, progression_rate=0.00005)
fault_manager.add_fault_trigger(
    "motor_insulation_degradation",
    conditions={"batteryTempMax_gt": 45, "motorTemperature_gt": 100, "sim_time_gt": 7200},
    severity_start=0.1,
    progression_rate=0.00001
)
fault_manager.add_fault_trigger("tire_pressure_loss", trigger_time_seconds=36000, progression_rate=0.000005)
fault_manager.add_fault_trigger("coolant_pump_failure", trigger_time_seconds=21600, progression_rate=0.0001)
fault_manager.add_fault_trigger(
    "sensor_frozen",
    trigger_time_seconds=7200,
    intermittent=True,
    intermittent_interval_s=120,
    intermittent_duration_s=10,
    details={"sensor": "motorTemperature"}
)
# Yeni arıza tetikleyicisi örnekleri (dashboard'dan tetiklenebilir)
# fault_manager.add_fault_trigger("sensor_offset", details={"sensor": "batterySOC", "offset": 5}, trigger_time_seconds=10000)
# fault_manager.add_fault_trigger("sensor_noisy", details={"sensor": "motorCurrent"}, trigger_time_seconds=12000)


# --- BAŞLANGIÇ DURUMU ---
current_speed_kph = 0
current_speed_mps = 0
total_distance_km = 0
charging_status = False
simulation_start_time = datetime.now()
last_print_time = time.time()


# --- FİZİKSEL HESAPLAMA YARDIMCI FONKSİYONLARI ---
def calculate_air_resistance(speed_mps, wind_speed_mps):
    effective_speed_mps = max(0, speed_mps + wind_speed_mps)
    return 0.5 * 1.225 * DRAG_COEFFICIENT * FRONTAL_AREA_SQM * (effective_speed_mps ** 2)

def calculate_rolling_resistance(mass_kg):
    return ROLLING_RESISTANCE_COEFF * mass_kg * GRAVITY

# --- GÖRSELLEŞTİRME KURULUMU (Python tarafı) ---
MAX_DATA_POINTS = 500
time_data = deque(maxlen=MAX_DATA_POINTS)
speed_data = deque(maxlen=MAX_DATA_POINTS)
soc_data = deque(maxlen=MAX_DATA_POINTS)
motor_temp_data = deque(maxlen=MAX_DATA_POINTS)
battery_temp_max_data = deque(maxlen=MAX_DATA_POINTS)
health_status_data = deque(maxlen=MAX_DATA_POINTS)
ambient_temp_data = deque(maxlen=MAX_DATA_POINTS)

HEALTH_MAP = {"normal_calisma": 0, "batarya_sicaklik_problemi": 1, "motor_izolasyon_problemi": 2,
              "lastik_basinci_dusuk": 3, "sogutma_pompasi_arizasi": 4, "yardimci_aku_performans_dusuk": 5,
              "motorTemperature_sensor_frozen": 6, "motorTemperature_sensor_offset": 7, "motorTemperature_sensor_noisy": 8}

# plt.style.use('ggplot')
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
# plt.ion()
# plt.show()

# def update_plot(bus_data):
#     # ... matplotlib plotting code (removed for now as it's not the focus)
#     pass # Placeholder

# --- HTTP KOMUT SUNUCUSU KISMI ---
# Aiohttp ile komutları işleyecek handler
async def handle_command(request):
    try:
        command = await request.json()
        print(f"HTTP sunucudan komut alındı: {command}")

        global simulation_start_time # Simülasyon zamanını kullanmak için global yaptık

        if command.get("type") == "set_driver_profile":
            driver.set_profile(command.get("profile"))
            return web.json_response({"status": "ok", "message": f"Driver profile set to {command.get('profile')}"})
        elif command.get("type") == "inject_fault":
            fault_type = command.get("fault_type")
            details = command.get("details")
            
            fault_manager.add_fault_trigger(
                fault_type=fault_type,
                severity_start=command.get("severity_start", 0.1),
                progression_rate=command.get("progression_rate", 0.0001),
                intermittent=command.get("intermittent", False),
                intermittent_interval_s=command.get("intermittent_interval_s", 60),
                intermittent_duration_s=command.get("intermittent_duration_s", 5),
                details=details,
                conditions=None,
                trigger_time_seconds=(datetime.now() - simulation_start_time).total_seconds() + 1 # 1 saniye sonra başlasın
            )
            return web.json_response({"status": "ok", "message": f"Fault {fault_type} injected"})
        elif command.get("type") == "clear_faults":
            fault_manager.active_faults.clear()
            fault_manager.triggered_fault_types.clear()
            fault_manager.sensor_override_faults.clear()
            print("--- TÜM ARIZALAR TEMİZLENDİ ---")
            return web.json_response({"status": "ok", "message": "All faults cleared"})
        else:
            return web.json_response({"status": "error", "message": "Unknown command"}, status=400)
    except Exception as e:
        print(f"HTTP komutunu işlerken hata: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)

async def start_command_server():
    app = web.Application()
    # CORS ayarları: Next.js uygulamanızın (localhost:3000) bu sunucuya bağlanmasına izin verir.
    # PROD ortamında bu *daha kısıtlı* olmalıdır.
    cors = web.middleware.cors.CorsMiddleware(
        allow_all=True, # Geliştirme için geçici olarak her şeye izin ver
        defaults={
            "*": web.middleware.cors.ResourceOptions(
                allow_credentials=True,
                allow_headers=("Content-Type",),
                allow_methods=("POST", "GET")
            )
        }
    )
    app.middlewares.append(cors) # CORS middleware'i ekle
    app.router.add_post('/command', handle_command) # /command adresine POST isteklerini dinle

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, COMMAND_SERVER_HOST, COMMAND_SERVER_PORT)
    await site.start()
    print(f"HTTP Komut Sunucusu {COMMAND_SERVER_HOST}:{COMMAND_SERVER_PORT} üzerinde dinliyor.")

# --- ANA SİMÜLASYON KODU (Async hale getirildi) ---
async def main_simulation_loop():
    global current_speed_kph, current_speed_mps, total_distance_km, charging_status, simulation_start_time, last_print_time

    last_update_time = time.perf_counter()
    # last_plot_update_time = time.time() # Artık kullanılmıyor

    while (datetime.now() - simulation_start_time).total_seconds() < MAX_SIMULATION_DURATION_SECONDS:
        current_time_perf = time.perf_counter()
        dt = current_time_perf - last_update_time
        last_update_time = current_time_perf
        
        if dt <= 0:
            dt = 1 / (1000 / SIMULATION_INTERVAL_MS)

        sim_time_elapsed_seconds = (datetime.now() - simulation_start_time).total_seconds()

        # --- 1. Rota ve Sürücü Kararları ---
        route_state = route_manager.update(current_speed_kph * dt / 3600)

        driver.set_target_speed(route_manager.current_speed_limit_kph)
        driver.set_traffic_density(route_manager.current_traffic_density)
        
        if route_manager.current_action == "charge_station":
            charging_status = True
            driver.set_target_speed(0)
            if current_speed_kph > 0.5:
                requested_accel_ms2 = -driver.current_profile_params["max_deccel_ms2"]
            else:
                requested_accel_ms2 = 0
                current_speed_kph = 0
                current_speed_mps = 0
        elif route_manager.current_action == "stop" or route_manager.current_action == "stop_traffic_light" or route_manager.current_action == "end_route":
            driver.set_target_speed(0)
            if current_speed_kph > 0.5:
                 requested_accel_ms2 = -driver.current_profile_params["max_deccel_ms2"]
            else:
                requested_accel_ms2 = 0
                current_speed_kph = 0
                current_speed_mps = 0
                charging_status = False
        elif route_manager.current_action == "end_of_route":
            driver.set_target_speed(0)
            requested_accel_ms2 = -driver.current_profile_params["max_deccel_ms2"] if current_speed_kph > 0 else 0
            current_speed_kph = 0
            current_speed_mps = 0
            charging_status = False
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Rota tamamlandı. Simülatör duruyor.")
            break
        else:
            charging_status = False
            requested_accel_ms2 = driver.get_acceleration_request(current_speed_kph, dt)


        # --- 2. Ortam Güncellemesi ---
        environment.update(dt, sim_time_elapsed_seconds)


        # --- 3. Hareket Dinamiği ve Güç Hesabı ---
        if not charging_status:
            F_air_resistance = calculate_air_resistance(current_speed_mps, environment.wind_speed_mps)
            F_rolling_resistance = calculate_rolling_resistance(MASS_KG)
            
            slope_angle_rad = np.radians(route_manager.current_slope_degrees)
            F_slope = MASS_KG * GRAVITY * np.sin(slope_angle_rad)

            F_net_request = (requested_accel_ms2 * MASS_KG) + F_air_resistance + F_rolling_resistance + F_slope

            motor_torque_request_nm = F_net_request * 0.5

            motor.update(motor_torque_request_nm, current_speed_kph, dt)
            
            motor_power_output_kw = (motor.current * motor.voltage / 1000) * motor.efficiency if motor.current > 0 else (motor.current * motor.voltage / 1000) * (1 / motor.efficiency)

            F_motor_actual = (motor_power_output_kw * 1000) / (current_speed_mps + 0.1) if current_speed_mps > 0 else 0
            
            actual_acceleration_ms2 = (F_motor_actual - F_air_resistance - F_rolling_resistance - F_slope) / MASS_KG
            
            current_speed_mps += actual_acceleration_ms2 * dt
            current_speed_mps = np.clip(current_speed_mps, 0, 120 / 3.6)
            current_speed_kph = current_speed_mps * 3.6

            total_distance_km += (current_speed_kph * dt) / 3600
            
            regen_brake_power = abs(motor_power_output_kw) if motor_power_output_kw < 0 else 0
            brake_pedal_active = True if requested_accel_ms2 < -0.5 else False
            if current_speed_kph == 0 and driver.target_speed_kph == 0:
                brake_pedal_active = True
            
        else:
            current_speed_kph = 0
            current_speed_mps = 0
            motor.rpm = 0
            motor.current = 0
            regen_brake_power = 0
            brake_pedal_active = True
            
        if current_speed_kph == 0:
            current_gear = "N" if charging_status else "P"
        else:
            current_gear = "D"

        # --- 4. Yardımcı Yükler (Dinamik Klima/Isıtma Yükü) ---
        cabin_temp_target = 22
        
        temp_diff = environment.ambient_temperature - cabin_temp_target
        hvac_power_kw = 0
        if abs(temp_diff) > 2:
            hvac_power_kw = (abs(temp_diff) * 0.8) + 5
            if environment.weather_condition in ["rainy", "snowy"] or environment.ambient_temperature > 35 or environment.ambient_temperature < 0:
                hvac_power_kw *= 1.2
        
        other_aux_power_kw = 2

        total_aux_power_kw = hvac_power_kw + other_aux_power_kw
        total_aux_current = (total_aux_power_kw * 1000) / battery.nominal_voltage

        # --- 5. Batarya Güncellemesi ---
        battery.update(motor.current, total_aux_current, dt)

        # --- 6. Sensör Verilerini Birleştirme (Gürültü apply_noise ile entegre edildi) ---
        bus_data = {
            "busId": BUS_ID,
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),
            "vehicleSpeed": round(apply_noise(current_speed_kph, "vehicleSpeed"), 1),
            "totalDistanceKm": round(total_distance_km, 2),
            "brakePedalActive": brake_pedal_active,
            "regenBrakePower": round(regen_brake_power, 1),
            "gear": current_gear,
            "auxBatteryVoltage": round(apply_noise(battery.nominal_voltage * (random.uniform(0.95, 1.05)), "auxBatteryVoltage"), 1),
            "cabinTemp": round(apply_noise(cabin_temp_target + random.uniform(-1, 1), "cabinTemp"), 1),
            "chargingStatus": charging_status,
            "tirePressure": round(apply_noise(80.0, "tirePressure"), 1) # Yeni sensör varsayımı
        }
        
        bus_data.update(motor.get_state())
        bus_data.update(battery.get_state())
        bus_data.update(environment.get_state())
        bus_data.update(route_manager.get_state())
        bus_data.update(driver.get_state())

        # --- 7. Arıza Yönetimi ve Etiketleme ---
        health_status, error_code = fault_manager.update(sim_time_elapsed_seconds, bus_data, dt)
        bus_data["healthStatus"] = health_status
        bus_data["errorCode"] = error_code

        # --- 8. Veriyi Next.js API'sine Gönder ---
        # Polling sistemine geri döndüğümüz için, Python hala MongoDB'ye veri basmalı.
        try:
            response = requests.post(
                NEXTJS_API_URL,
                data=json.dumps(bus_data, cls=NumpyEncoder),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] MongoDB'ye VERİ GÖNDERME HATASI: {e}")

        # --- 9. Konsol ve Python Görselleştirme Güncelleme ---
        if (time.time() - last_print_time) >= 1 or bus_data["healthStatus"] != "normal_calisma" or bus_data["chargingStatus"]:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
                    f"Hız: {bus_data['vehicleSpeed']} km/s, "
                    f"SOC: {bus_data['batterySOC']}%, "
                    f"Motor Sıcaklığı: {bus_data['motorTemperature']}°C, "
                    f"Batarya Max Sıcaklık: {bus_data['batteryTempMax']}°C, "
                    f"Dış Sıcaklık: {bus_data['ambientTemperature']}°C, "
                    f"Hava: {bus_data['weatherCondition']}, "
                    f"Durum: {bus_data['healthStatus']}{' (Şarjda)' if bus_data['chargingStatus'] else ''}"
                    f" (Sürücü: {bus_data['driverProfile']}"
                    f"{', Hız Sabitleyici: Aktif' if bus_data['cruiseControlActive'] else ''})")
            last_print_time = time.time()
        
        # Python tarafı görselleştirmeyi kapatıyoruz.
        # if random.random() < 0.1:
        #    update_plot(bus_data)
        
        # Yapay bekleme süresini tamamen kaldırdığımız için, CPU'nun elverdiği hızda çalışır.
        # await asyncio.sleep(0) # Bu, kontrolü event loop'a geri vermek için çok kısa bir bekleme sağlar.


async def main():
    # Komut sunucusunu başlat
    command_server_task = asyncio.create_task(start_command_server())

    # Simülasyon döngüsünü başlat
    simulation_task = asyncio.create_task(main_simulation_loop())

    # İki görevi eş zamanlı olarak çalıştır
    await asyncio.gather(command_server_task, simulation_task)


if __name__ == "__main__":
    try:
        print("Ana program başlatılıyor...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nElektrikli Otobüs Simülatörü Durduruldu.")
        # plt.close('all') # Eğer matplotlib penceresi açmıyorsanız bu satırı kaldırabilirsiniz
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")
        # plt.close('all') # Eğer matplotlib penceresi açmıyorsanız bu satırı kaldırabilirsiniz