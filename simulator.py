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
from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions

# components dosyasından sınıfları içe aktar
from electric_bus_components import ElectricMotor, Battery, Driver, Environment, FaultManager, RouteManager, apply_noise, calculate_resistances

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
        elif isinstance(obj, np.ndarray):
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
SIMULATION_INTERVAL_MS = 1 # Her 1ms'de bir simülasyon adımı (hızlı)
MAX_SIMULATION_DURATION_SECONDS = 3600 * 24 * 5 # 5 günlük simülasyon

# --- OTOBÜS FİZİKSEL SABİTLERİ ---
MASS_KG = 18000
DRAG_COEFFICIENT = 0.6
FRONTAL_AREA_SQM = 7.0
ROLLING_RESISTANCE_COEFF = 0.01
GRAVITY = 9.81 # Yerçekimi ivmesi (m/s^2)

# --- BATARYA ŞARJ AYARLARI ---
CHARGING_RATE_KW = 250 # kW cinsinden şarj gücü (örneğin 250 kW hızlı şarj)

# --- SİMÜLASYON NESNELERİNİ BAŞLAT (Önce rota verisi tanımlanmalı) ---

# --- ÖRNEK ROTA VERİSİ (Adana - İstanbul Daha Detaylı Senaryo) ---
EXAMPLE_ROUTE_DATA = [
    # --- Adana'dan Çıkış (Otobüsün doğrudan hızlanmaya başlamasını sağlayacak şekilde güncellendi) ---
    {"distance_km": 0.005, "slope_degrees": 0, "speed_limit_kph": 80, "traffic_density": "low", "action": "initial_burst_acceleration", "initial_bearing": 330}, # 5 metrede 80km/s'ye çıkmaya zorla
    {"distance_km": 5, "slope_degrees": 0, "speed_limit_kph": 60, "traffic_density": "low", "action": "city_start_cruise", "initial_bearing": 330}, # Sonra normal seyre geç
    {"distance_km": 10, "slope_degrees": 0, "speed_limit_kph": 70, "traffic_density": "medium", "action": "drive_city_exit", "initial_bearing": 330},
    {"distance_km": 50, "slope_degrees": 0.5, "speed_limit_kph": 100, "traffic_density": "low", "action": "drive_highway_start", "initial_bearing": 330},

    # --- Niğde / Pozantı Tüneli Yaklaşımı ---
    {"distance_km": 30, "slope_degrees": 3, "speed_limit_kph": 70, "traffic_density": "low", "action": "drive_mountain_uphill", "initial_bearing": 320},
    {"distance_km": 20, "slope_degrees": -2, "speed_limit_kph": 85, "traffic_density": "low", "action": "drive_mountain_downhill", "initial_bearing": 325},
    {"distance_km": 5, "slope_degrees": 0, "speed_limit_kph": 0, "traffic_density": "low", "action": "stop_rest_area", "initial_bearing": 330},

    # --- Kapadokya Bölgesi / Kayseri Yönü (Basitleştirilmiş) ---
    {"distance_km": 1.0, "slope_degrees": 0, "speed_limit_kph": 0, "traffic_density": "low", "action": "charge_station", "initial_bearing": 0}, # Şarj durağı
    {"distance_km": 80, "slope_degrees": 1, "speed_limit_kph": 90, "traffic_density": "low", "action": "drive_long_haul_plateau", "initial_bearing": 340},
    {"distance_km": 70, "slope_degrees": -0.5, "speed_limit_kph": 100, "traffic_density": "low", "action": "drive_long_haul_downhill", "initial_bearing": 335},

    # --- Ankara Çevresi (Geçiş) ---
    {"distance_km": 10, "slope_degrees": 0, "speed_limit_kph": 60, "traffic_density": "medium", "action": "drive_ankara_bypass", "initial_bearing": 300},
    {"distance_km": 5, "slope_degrees": 0, "speed_limit_kph": 0, "traffic_density": "high", "action": "stop_traffic_jam", "initial_bearing": 300},
    {"distance_km": 40, "slope_degrees": 0, "speed_limit_kph": 90, "traffic_density": "low", "action": "drive_after_traffic", "initial_bearing": 290},

    # --- Bolu Dağı / Tüneli Yaklaşımı ---
    {"distance_km": 25, "slope_degrees": 2.5, "speed_limit_kph": 80, "traffic_density": "low", "action": "drive_bolu_uphill", "initial_bearing": 280},
    {"distance_km": 20, "slope_degrees": -2.0, "speed_limit_kph": 85, "traffic_density": "low", "action": "drive_bolu_downhill", "initial_bearing": 285},
    
    # --- Kocaeli / İzmit Geçişi ---
    {"distance_km": 60, "slope_degrees": 0, "speed_limit_kph": 100, "traffic_density": "low", "action": "drive_izm_kocaeli", "initial_bearing": 290},
    {"distance_km": 5, "slope_degrees": 0, "speed_limit_kph": 0, "traffic_density": "medium", "action": "stop_toll_gate", "initial_bearing": 290},

    # --- İstanbul'a Giriş ---
    {"distance_km": 30, "slope_degrees": 0, "speed_limit_kph": 70, "traffic_density": "high", "action": "drive_istanbul_entry", "initial_bearing": 280},
    {"distance_km": 10, "slope_degrees": 0, "speed_limit_kph": 40, "traffic_density": "high", "action": "drive_istanbul_traffic", "initial_bearing": 275},
    {"distance_km": 0.1, "slope_degrees": 0, "speed_limit_kph": 0, "traffic_density": "high", "action": "end_of_route", "initial_bearing": 0}
]

# SİMÜLASYON NESNELERİNİ BAŞLAT (Rota verisi tanımlandıktan sonra)
motor = ElectricMotor()
battery = Battery()
environment = Environment(initial_temp=25)
fault_manager = FaultManager()

# Sürücü ve Rota Yöneticisini, EXAMPLE_ROUTE_DATA tanımlandıktan sonra başlatın
driver = Driver(profile="normal", target_speed_kph=EXAMPLE_ROUTE_DATA[0]["speed_limit_kph"])
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


# --- BAŞLANGIÇ DURUMU ---
current_speed_kph = 0
current_speed_mps = 0
total_distance_km = 0
charging_status = False
simulation_start_time = datetime.now()
last_print_time = time.time()


# --- HTTP KOMUT SUNUCUSU KISMI ---
async def handle_command(request):
    try:
        command = await request.json()
        print(f"HTTP sunucudan komut alındı: {command}")

        global simulation_start_time # Simülasyon zamanını kullanmak için global yaptık
        global CHARGING_RATE_KW # CHARGING_RATE_KW'yi de global yapalım ki komutlar etkilesin

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
        elif command.get("type") == "set_charging_rate":
            new_rate = command.get("rate_kw")
            if isinstance(new_rate, (int, float)) and new_rate >= 0:
                CHARGING_RATE_KW = new_rate
                print(f"Şarj oranı güncellendi: {CHARGING_RATE_KW} kW")
                return web.json_response({"status": "ok", "message": f"Charging rate set to {CHARGING_RATE_KW} kW"})
            else:
                return web.json_response({"status": "error", "message": "Invalid charging rate"}, status=400)
        else:
            return web.json_response({"status": "error", "message": "Unknown command"}, status=400)
    except Exception as e:
        print(f"HTTP komutunu işlerken hata: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)

async def start_command_server():
    app = web.Application()
    cors = cors_setup(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            allow_headers=("Content-Type",),
            allow_methods=["POST", "GET"]
        )
    })
    for route in list(app.router.routes()):
        cors.add(route)

    app.router.add_post('/command', handle_command)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, COMMAND_SERVER_HOST, COMMAND_SERVER_PORT)
    await site.start()
    print(f"HTTP Komut Sunucusu {COMMAND_SERVER_HOST}:{COMMAND_SERVER_PORT} üzerinde dinliyor.")

# --- ANA SİMÜLASYON KODU (Async hale getirildi) ---
async def main_simulation_loop():
    global current_speed_kph, current_speed_mps, total_distance_km, charging_status, simulation_start_time, last_print_time

    last_update_time = time.perf_counter()

    while (datetime.now() - simulation_start_time).total_seconds() < MAX_SIMULATION_DURATION_SECONDS:
        current_time_perf = time.perf_counter()
        dt = current_time_perf - last_update_time
        last_update_time = current_time_perf
        
        if dt <= 0:
            dt = 1 / (1000 / SIMULATION_INTERVAL_MS)

        sim_time_elapsed_seconds = (datetime.now() - simulation_start_time).total_seconds()
        # Kaç dakikadır yolda olduğunu hesapla
        minutes_on_road = round(sim_time_elapsed_seconds / 60)

        # Hangi şehirde olduğunu takip et (basit mesafe tabanlı mantık)
        current_city = "Adana"
        if total_distance_km > 100: # Niğde civarı
            current_city = "Niğde"
        if total_distance_km > 250: # Ankara civarı
            current_city = "Ankara"
        if total_distance_km > 450: # Bolu civarı
            current_city = "Bolu"
        if total_distance_km > 550: # Kocaeli civarı
            current_city = "Kocaeli"
        if total_distance_km > 650: # İstanbul civarı
            current_city = "İstanbul"


        # --- 1. Rota ve Sürücü Kararları ---
        route_state = route_manager.update(current_speed_kph * dt / 3600)

        driver.set_target_speed(route_manager.current_speed_limit_kph)
        driver.set_traffic_density(route_manager.current_traffic_density)
        
        if route_manager.current_action == "charge_station":
            charging_status = True
            driver.set_target_speed(0)
            
            if current_speed_kph > 0.5:
                requested_accel_ms2 = -driver.current_profile_params["max_deccel_ms2"]
                current_speed_mps += requested_accel_ms2 * dt
                current_speed_mps = np.clip(current_speed_mps, 0, 120 / 3.6)
                current_speed_kph = current_speed_mps * 3.6
                motor_current = 0
                regen_brake_power = 0
            else:
                current_speed_kph = 0
                current_speed_mps = 0
                requested_accel_ms2 = 0
                regen_brake_power = 0
                brake_pedal_active = True
                motor_current = 0

                if battery.soc < 95:
                    charging_current_amps = -(CHARGING_RATE_KW * 1000) / battery.nominal_voltage
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] >>> Şarj Ediliyor: {CHARGING_RATE_KW} kW, SOC: {battery.soc:.1f}% <<<")
                else:
                    charging_status = False
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] >>> Şarj Tamamlandı, SOC: {battery.soc:.1f}% <<<")
                    route_manager.distance_in_current_segment_km = route_manager.route_data[route_manager.current_segment_index]["distance_km"]
                    continue
            
            total_aux_current_for_charging = (hvac_power_kw + other_aux_power_kw) * 1000 / battery.nominal_voltage
            battery.update(motor_current + total_aux_current_for_charging + charging_current_amps, dt)
            
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
            # Simülasyonu burada bitirmek için 'break'i yorum satırından çıkarabilirsiniz.
        
        else:
            charging_status = False
            requested_accel_ms2 = driver.get_desired_acceleration(current_speed_kph, dt)


        # --- 2. Ortam Güncellemesi ---
        environment.update(dt, sim_time_elapsed_seconds)


        # --- 3. Hareket Dinamiği ve Güç Hesabı (Kinematik Odaklı) ---
        if not charging_status:
            current_speed_mps += requested_accel_ms2 * dt
            current_speed_mps = np.clip(current_speed_mps, 0, 120 / 3.6)
            current_speed_kph = current_speed_mps * 3.6

            distance_traveled_this_step_km = (current_speed_kph * dt) / 3600 # <-- Debug için
            total_distance_km += distance_traveled_this_step_km # <-- totalDistanceKm güncelleniyor
            
            # DEBUG çıktıları:
            # print(f"DEBUG_DIST: Speed: {current_speed_kph:.2f} km/h, dt: {dt:.4f} s, Dist this step: {distance_traveled_this_step_km:.6f} km, Total Dist: {total_distance_km:.6f} km")


            F_air_resistance, F_rolling_resistance, F_slope = calculate_resistances(
                current_speed_mps, route_manager.current_slope_degrees, MASS_KG, DRAG_COEFFICIENT,
                FRONTAL_AREA_SQM, ROLLING_RESISTANCE_COEFF, environment.wind_speed_mps
            )

            F_traction_required_total = (requested_accel_ms2 * MASS_KG) + F_air_resistance + F_rolling_resistance + F_slope
            
            motor_power_output_kw, motor_current = motor.calculate_power_and_current(
                F_traction_required_total, current_speed_mps, dt
            )
            
            regen_brake_power = abs(motor_power_output_kw) if motor_power_output_kw < 0 else 0
            brake_pedal_active = True if requested_accel_ms2 < -0.5 else False
            if current_speed_kph == 0 and driver.target_speed_kph == 0:
                brake_pedal_active = True
            
        else:
            motor_current = 0
            regen_brake_power = 0
            brake_pedal_active = True
            current_speed_kph = 0 # Şarjdayken hız 0
            current_speed_mps = 0


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
        if not charging_status: # Sadece şarj olmuyorsa motor akımı ve yardımcı yükleri bataryaya gönder
            battery.update(motor_current + total_aux_current, dt)
        # Eğer şarj oluyorsa, batarya zaten yukarıdaki "charge_station" bloğunda şarj akımıyla güncelleniyor.

        # --- 6. Sensör Verilerini Birleştirme ---
        bus_data = {
            "busId": BUS_ID,
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),
            "vehicleSpeed": round(apply_noise(current_speed_kph, "vehicleSpeed"), 1),
            "totalDistanceKm": round(total_distance_km, 2), # <-- totalDistanceKm burada güncellenmiş haliyle alınır
            "brakePedalActive": brake_pedal_active,
            "regenBrakePower": round(regen_brake_power, 1),
            "gear": current_gear,
            "auxBatteryVoltage": round(apply_noise(battery.nominal_voltage * (random.uniform(0.95, 1.05)), "auxBatteryVoltage"), 1),
            "cabinTemp": round(apply_noise(cabin_temp_target + random.uniform(-1, 1), "cabinTemp"), 1),
            "chargingStatus": charging_status,
            "tirePressure": round(apply_noise(80.0, "tirePressure"), 1)
        }
        
        bus_data.update(motor.get_state())
        bus_data.update(battery.get_state())
        bus_data.update(environment.get_state())
        bus_data.update(route_manager.get_state())
        bus_data.update(driver.get_state())

        # --- Yeni Eklenen Alanlar ---
        bus_data["minutesOnRoad"] = minutes_on_road # <-- Yeni
        bus_data["currentCity"] = current_city # <-- Yeni

        # --- 7. Arıza Yönetimi ve Etiketleme ---
        health_status, error_code = fault_manager.update(sim_time_elapsed_seconds, bus_data, dt)
        bus_data["healthStatus"] = health_status
        bus_data["errorCode"] = error_code

        # --- 8. Veriyi Next.js API'sine Gönder ---
        try:
            response = requests.post(
                NEXTJS_API_URL,
                data=json.dumps(bus_data, cls=NumpyEncoder),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] MongoDB'ye VERİ GÖNDERME HATASI: {e}")

        # --- 9. Konsol Çıktısı ---
        if (time.time() - last_print_time) >= 1 or bus_data["healthStatus"] != "normal_calisma" or bus_data["chargingStatus"]:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
                    f"Hız: {bus_data['vehicleSpeed']} km/s, "
                    f"SOC: {bus_data['batterySOC']}%, "
                    f"Konum: ({bus_data['latitude']:.6f},{bus_data['longitude']:.6f}), "
                    f"Yön: {bus_data['bearing_degrees']}°, "
                    f"Mesafe: {bus_data['totalDistanceKm']:.2f} km, " # <-- Mesafe eklendi
                    f"Yolda: {bus_data['minutesOnRoad']} dk, " # <-- Dakika eklendi
                    f"Şehir: {bus_data['currentCity']}, " # <-- Şehir eklendi
                    f"Durum: {bus_data['healthStatus']}{' (Şarjda)' if bus_data['chargingStatus'] else ''}"
                    f" (Sürücü: {bus_data['driverProfile']}), "
                    f"İstenen İvme: {requested_accel_ms2:.4f} m/s^2")
            last_print_time = time.time()
        
        # Yapay bekleme süresini tamamen kaldırdığımız için, CPU'nun elverdiği hızda çalışır.


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
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")