import numpy as np
import random
from datetime import datetime
import math
import time # <-- BU SATIR EKLENDİ

# --- Global Sensor Noise Parameters ---
DEFAULT_NOISE_STD_DEV = {
    "vehicleSpeed": 0.5, # km/s
    "motorRPM": 5,   # Kapalıyken daha az gürültü
    "motorCurrent": 2, # Kapalıyken daha az gürültü
    "motorVoltage": 0.1, # Kapalıyken daha az gürültü
    "motorTemperature": 0.5,
    "batterySOC": 0.1,
    "batteryVoltage": 0.2,
    "batteryCurrent": 3,
    "batteryTempMin": 0.3,
    "batteryTempMax": 0.3,
    "batteryHealth": 0.01,
    "auxBatteryVoltage": 0.1,
    "cabinTemp": 0.2,
    "ambientTemperature": 0.3,
    "windSpeedMps": 0.5,
    "windDirectionDegrees": 5,
    "current_slope_degrees": 0.1,
    "tirePressure": 0.5,
    "latitude": 0.000001,
    "longitude": 0.000001
}

# --- Function to apply noise ---
def apply_noise(value, sensor_name):
    std_dev = DEFAULT_NOISE_STD_DEV.get(sensor_name, 0)
    if value == 0 and std_dev > 0 and sensor_name in ["motorRPM", "motorCurrent", "motorVoltage"]:
        return np.random.normal(0, std_dev * 0.1)
    return value + np.random.normal(0, std_dev)

# --- Yardımcı Fonksiyon: Mesafe ve Yöne Göre Yeni GPS Koordinatları Hesaplama ---
def calculate_new_coords(lat, lon, bearing_degrees, distance_km):
    R = 6371
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_degrees)

    lat2_rad = math.asin(math.sin(lat_rad) * math.cos(distance_km / R) +
                         math.cos(lat_rad) * math.sin(distance_km / R) * math.cos(bearing_rad))
    lon2_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat_rad),
                                   math.cos(distance_km / R) - math.sin(lat_rad) * math.sin(lat2_rad))

    new_lat = math.degrees(lat2_rad)
    new_lon = math.degrees(lon2_rad) 
    return new_lat, new_lon

# --- Yardımcı Fonksiyon: Direnç Kuvvetleri Hesaplama (Rüzgar Yönü ve Yol Koşulu Eklendi) ---
def calculate_resistances(speed_mps, slope_degrees, mass_kg, drag_coefficient, frontal_area_sqm, rolling_resistance_coeff, wind_speed_mps, wind_direction_degrees, vehicle_bearing_degrees, weather_condition):
    GRAVITY = 9.81
    air_density = 1.225

    angle_diff = abs(vehicle_bearing_degrees - wind_direction_degrees)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    headwind_component_factor = math.cos(math.radians(angle_diff))
    effective_wind_speed = wind_speed_mps * headwind_component_factor
    
    effective_speed_mps_for_air = max(0, speed_mps + effective_wind_speed)
    F_air_resistance = 0.5 * air_density * drag_coefficient * frontal_area_sqm * (effective_speed_mps_for_air ** 2)

    adjusted_rolling_resistance_coeff = rolling_resistance_coeff
    if weather_condition == "rainy":
        adjusted_rolling_resistance_coeff *= 1.2
    elif weather_condition == "snowy":
        adjusted_rolling_resistance_coeff *= 1.5

    F_rolling_resistance = adjusted_rolling_resistance_coeff * mass_kg * GRAVITY

    slope_angle_rad = math.radians(slope_degrees)
    F_slope = mass_kg * GRAVITY * math.sin(slope_angle_rad)

    return F_air_resistance, F_rolling_resistance, F_slope

# --- Otobüs Konfigürasyon Sınıfı ---
class BusConfig:
    def __init__(self,
                 model_name="TEMSA TS45E",
                 mass_kg=24494,
                 drag_coefficient=0.6,
                 frontal_area_sqm=7.0,
                 rolling_resistance_coeff=0.01,
                 max_motor_power_kw=372,
                 motor_efficiency=0.92,
                 battery_capacity_kwh=560,
                 battery_nominal_voltage=650,
                 battery_internal_resistance=0.0005,
                 charging_rate_kw=150
                 ):
        self.model_name = model_name
        self.mass_kg = mass_kg
        self.drag_coefficient = drag_coefficient
        self.frontal_area_sqm = frontal_area_sqm
        self.rolling_resistance_coeff = rolling_resistance_coeff
        self.max_motor_power_kw = max_motor_power_kw
        self.motor_efficiency = motor_efficiency
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_nominal_voltage = battery_nominal_voltage
        self.battery_internal_resistance = battery_internal_resistance
        self.charging_rate_kw = charging_rate_kw


class ElectricMotor:
    def __init__(self, config: BusConfig):
        self.max_power_kw = config.max_motor_power_kw
        self.efficiency = config.motor_efficiency
        self.nominal_voltage = config.battery_nominal_voltage

        self.rpm = 0
        self.current = 0
        self.voltage = self.nominal_voltage
        self.temperature = 30
        self.degradation_factor = 1.0
        self.status = "on"

    def calculate_power_and_current(self, required_traction_force_newtons, speed_mps, dt):
        if self.status == "off":
            self.current = 0
            self.rpm = 0
            power_output_kw = 0
            self.temperature += -(self.temperature - 30) * 0.05 * dt + random.uniform(-0.05, 0.05)
            self.temperature = np.clip(self.temperature, 20, 150)
            return power_output_kw, self.current
            
        power_output_watts = required_traction_force_newtons * speed_mps
        power_output_kw = power_output_watts / 1000

        effective_efficiency = self.efficiency / self.degradation_factor

        if power_output_kw > 0:
            self.current = (power_output_kw * 1000) / (self.nominal_voltage * effective_efficiency)
        elif power_output_kw < 0:
            self.current = (power_output_kw * 1000) / (self.nominal_voltage * (1 / effective_efficiency))
        else:
            self.current = 0

        max_possible_current = self.max_power_kw * 1000 / self.nominal_voltage
        self.current = np.clip(self.current, -max_possible_current, max_possible_current)

        self.rpm = min(5000, max(0, speed_mps * 30))

        heat_generated_kw = (abs(self.current) * self.nominal_voltage * (1 - self.efficiency)) / 1000
        self.temperature += (heat_generated_kw * 0.1 - (self.temperature - 30) * 0.05) * dt + random.uniform(-0.05, 0.05)
        self.temperature = np.clip(self.temperature, 20, 150)

        return power_output_kw, self.current

    def get_state(self):
        reported_rpm = self.rpm
        reported_current = self.current
        reported_voltage = self.voltage

        if self.status == "off":
            reported_rpm = 0
            reported_current = 0
            reported_voltage = self.nominal_voltage

        return {
            "motorRPM": round(apply_noise(reported_rpm, "motorRPM")),
            "motorCurrent": round(apply_noise(reported_current, "motorCurrent"), 1),
            "motorVoltage": round(apply_noise(reported_voltage, "motorVoltage"), 1),
            "motorTemperature": round(apply_noise(self.temperature, "motorTemperature"), 1),
            "motorStatus": self.status
        }

class Battery:
    def __init__(self, config: BusConfig):
        self.capacity_kwh = config.battery_capacity_kwh
        self.nominal_voltage = config.battery_nominal_voltage
        self.internal_resistance = config.battery_internal_resistance

        self.soc = 80
        self.voltage = self.nominal_voltage
        self.current = 0
        self.temp_min = 25
        self.temp_max = 28
        self.health = 100

        self.battery_capacity_ah = (self.capacity_kwh * 1000) / self.nominal_voltage

        self.degradation_factor = 1.0
        self.bms_fault_active = False

    def update(self, total_current_amps, dt):
        self.current = total_current_amps

        effective_internal_resistance = self.internal_resistance * self.degradation_factor
        self.voltage = self.nominal_voltage - (self.current * effective_internal_resistance)
        self.voltage = np.clip(self.voltage, self.nominal_voltage * 0.8, self.nominal_voltage * 1.1)

        coulombs_transferred = self.current * dt
        effective_capacity_ah = self.battery_capacity_ah / self.degradation_factor
        soc_change_percent = (coulombs_transferred / (effective_capacity_ah * 3600)) * 100

        self.soc -= soc_change_percent
        self.soc = np.clip(self.soc, 0, 100)

        heat_generated = (abs(self.current) ** 2 * effective_internal_resistance) / 1000
        self.temp_min += (heat_generated * 0.2 - (self.temp_min - 25) * 0.02) * dt + random.uniform(-0.02, 0.02)
        self.temp_max += (heat_generated * 0.2 - (self.temp_max - 28) * 0.02) * dt + random.uniform(-0.02, 0.02)
        
        self.temp_min = np.clip(self.temp_min, 10, 70)
        self.temp_max = np.clip(self.temp_max, 10, 70)

        self.health = np.clip(self.health, 0, 100)

    def get_state(self):
        return {
            "batterySOC": round(apply_noise(self.soc, "batterySOC"), 1),
            "batteryVoltage": round(apply_noise(self.voltage, "batteryVoltage"), 1),
            "batteryCurrent": round(apply_noise(self.current, "batteryCurrent"), 1),
            "batteryTempMin": round(apply_noise(self.temp_min, "batteryTempMin"), 1),
            "batteryTempMax": round(apply_noise(self.temp_max, "batteryTempMax"), 1),
            "batteryHealth": round(apply_noise(self.health, "batteryHealth"), 2),
            "bmsFaultActive": self.bms_fault_active
        }

class Driver:
    def __init__(self, profile="normal", target_speed_kph=0, reaction_delay_seconds=0.1):
        self.profile = profile
        self.target_speed_kph = target_speed_kph
        self.reaction_delay_seconds = reaction_delay_seconds
        self._last_target_speed_change_time = 0

        self.profiles = {
            "normal": {
                "max_accel_ms2": 1.5,
                "max_deccel_ms2": 2.5,
                "reaction_factor": 0.05,
                "speed_limit_adherence": 1.0
            },
            "aggressive": {
                "max_accel_ms2": 2.5,
                "max_deccel_ms2": 4.0,
                "reaction_factor": 0.1,
                "speed_limit_adherence": 1.1
            },
            "defensive": {
                "max_accel_ms2": 1.0,
                "max_deccel_ms2": 1.5,
                "reaction_factor": 0.03,
                "speed_limit_adherence": 0.9
            },
            "tired": {
                "max_accel_ms2": 1.2,
                "max_deccel_ms2": 3.0,
                "reaction_factor": 0.02,
                "speed_limit_adherence": 1.0
            }
        }
        self.current_profile_params = self.profiles.get(self.profile, self.profiles["normal"])

        self.cruise_control_active = False
        self.cruise_control_speed_kph = 0
        self.traffic_density = "low"
        self.stop_and_go_timer = 0
        self.stop_and_go_interval = random.uniform(5, 15)
        self.go_interval = random.uniform(5, 10)

    def set_profile(self, new_profile):
        if new_profile in self.profiles:
            self.profile = new_profile
            self.current_profile_params = self.profiles[new_profile]
            print(f"Sürücü profili '{self.profile}' olarak ayarlandı.")
        else:
            print(f"Uyarı: Bilinmeyen sürücü profili '{new_profile}'. 'normal' profile geçiliyor.")
            self.profile = "normal"
            self.current_profile_params = self.profiles["normal"]

    def set_target_speed(self, new_speed):
        if new_speed != self.target_speed_kph:
            self._last_target_speed_change_time = time.perf_counter()
        self.target_speed_kph = new_speed

    def set_traffic_density(self, density):
        self.traffic_density = density

    def toggle_cruise_control(self, activate, current_speed_kph=0):
        if activate and current_speed_kph > 20:
            self.cruise_control_active = True
            self.cruise_control_speed_kph = current_speed_kph
            print(f"Hız Sabitleyici Aktif: {round(self.cruise_control_speed_kph)} km/s.")
        else:
            self.cruise_control_active = False
            self.cruise_control_speed_kph = 0
            if activate: print("Hız Sabitleyici Devre Dışı Bırakıldı.")

    def get_desired_acceleration(self, current_speed_kph, dt):
        time_since_last_target_change = time.perf_counter() - self._last_target_speed_change_time
        if time_since_last_target_change < self.reaction_delay_seconds:
            if current_speed_kph < 1 and self.target_speed_kph > 0:
                return self.current_profile_params["max_accel_ms2"] * 0.1
            return 0

        target_speed_for_driver = self.target_speed_kph

        if self.cruise_control_active:
            target_speed_for_driver = self.cruise_control_speed_kph
            if abs(target_speed_for_driver - current_speed_kph) > 5:
                self.toggle_cruise_control(False)
        
        target_speed_for_driver *= self.current_profile_params["speed_limit_adherence"]
        target_speed_for_driver = min(target_speed_for_driver, self.target_speed_kph * 1.2)

        effective_target_speed_for_traffic = target_speed_for_driver
        
        if self.traffic_density == "high":
            effective_target_speed_for_traffic = min(effective_target_speed_for_traffic, 30)
            if current_speed_kph < effective_target_speed_for_traffic * 0.5:
                self.stop_and_go_timer += dt
                if self.stop_and_go_timer >= self.stop_and_go_interval:
                    self.stop_and_go_timer = 0
                    self.stop_and_go_interval = random.uniform(5, 15)
                    return self.current_profile_params["max_accel_ms2"] * random.uniform(0.7, 1.0)
                return -self.current_profile_params["max_deccel_ms2"] * random.uniform(0.1, 0.3)
            
        elif self.traffic_density == "medium":
            effective_target_speed_for_traffic = min(effective_target_speed_for_traffic, 70)
            
        error_speed = effective_target_speed_for_traffic - current_speed_kph
        
        desired_acceleration_ms2 = 0
        if abs(error_speed) > 0.01:
            if error_speed > 0:
                desired_acceleration_ms2 = error_speed * self.current_profile_params["reaction_factor"]
                desired_acceleration_ms2 = min(self.current_profile_params["max_accel_ms2"], desired_acceleration_ms2)
                
                if current_speed_kph < 5 and target_speed_for_driver > 0:
                    desired_acceleration_ms2 = max(desired_acceleration_ms2, self.current_profile_params["max_accel_ms2"] * 0.7)

            else: # Yavaşlama ihtiyacı (error_speed negatif)
                desired_acceleration_ms2 = error_speed * self.current_profile_params["reaction_factor"]
                desired_acceleration_ms2 = max(-self.current_profile_params["max_deccel_ms2"], desired_acceleration_ms2)
        
        desired_acceleration_ms2 += random.uniform(-0.001, 0.001)

        return desired_acceleration_ms2

    def get_state(self):
        return {
            "driverProfile": self.profile,
            "cruiseControlActive": self.cruise_control_active,
            "driverTargetSpeed": self.target_speed_kph
        }

class Environment:
    def __init__(self, initial_temp=25, initial_weather="clear", initial_wind_speed=5, initial_wind_direction=0):
        self.ambient_temperature = initial_temp
        self.weather_condition = initial_weather
        self.wind_speed_mps = initial_wind_speed
        self.humidity = 60
        self.wind_direction_degrees = initial_wind_direction

    def update(self, dt, current_sim_time_seconds):
        seconds_in_day = 24 * 3600
        time_of_day_fraction = (current_sim_time_seconds % seconds_in_day) / seconds_in_day
        
        min_temp = 10
        max_temp = 35
        
        daily_temp_swing = (max_temp - min_temp) / 2
        avg_daily_temp = (max_temp + min_temp) / 2
        
        self.ambient_temperature = avg_daily_temp + daily_temp_swing * np.sin(2 * np.pi * (time_of_day_fraction - 0.25))
        
        self.ambient_temperature += random.uniform(-0.5, 0.5) * dt
        self.ambient_temperature = np.clip(self.ambient_temperature, -15, 45)

        if random.random() < 0.000005 * dt:
            self.weather_condition = random.choice(["clear", "rainy", "snowy"])

        self.wind_speed_mps += random.uniform(-0.1, 0.1) * dt
        self.wind_speed_mps = np.clip(self.wind_speed_mps, 0, 25)

        self.wind_direction_degrees += random.uniform(-10, 10) * dt
        self.wind_direction_degrees = self.wind_direction_degrees % 360
        if self.wind_direction_degrees < 0: self.wind_direction_degrees += 360

    def get_state(self):
        return {
            "ambientTemperature": round(apply_noise(self.ambient_temperature, "ambientTemperature"), 1),
            "weatherCondition": self.weather_condition,
            "windSpeedMps": round(apply_noise(self.wind_speed_mps, "windSpeedMps"), 1),
            "windDirectionDegrees": round(apply_noise(self.wind_direction_degrees, "windDirectionDegrees"), 1),
            "humidity": round(apply_noise(self.humidity, "humidity"), 1)
        }

class FaultManager:
    def __init__(self):
        self.active_faults = []
        self.fault_triggers = []
        self.triggered_fault_types = set()
        self.sensor_override_faults = {} # {"sensor_name": {"type": "frozen"/"offset", "value": X}}

    def add_fault_trigger(self, fault_type, trigger_time_seconds=None, conditions=None, severity_start=0.1, progression_rate=0.0001, intermittent=False, intermittent_interval_s=60, intermittent_duration_s=5, details=None):
        """
        Yeni bir arıza tetikleyicisi ekler.
        :param fault_type: Arızanın tipi
        :param trigger_time_seconds: Zaman tabanlı tetikleyici
        :param conditions: Koşul tabanlı tetikleyici
        :param severity_start: Başlangıç şiddeti
        :param progression_rate: Kötüleşme hızı
        :param intermittent: Arıza aralıklı mı olacak (True/False)
        :param intermittent_interval_s: Aralıklı arızanın tekrar etme aralığı (saniye)
        :param intermittent_duration_s: Aralıklı arızanın ne kadar sürdüğü (saniye)
        :param details: Sensör arızaları gibi özel detaylar için (örn: {"sensor": "motorTemperature", "offset": 10})
        """
        if fault_type in self.triggered_fault_types and not intermittent:
            print(f"Uyarı: '{fault_type}' arızası zaten tanımlı ve tetiklenmiş durumda, tekrar eklenmiyor.")
            return

        if fault_type.startswith("sensor_") and details and details.get("sensor") in self.sensor_override_faults and not intermittent:
            print(f"Uyarı: '{fault_type}' ({details.get('sensor')}) arızası zaten aktif durumda, tekrar eklenmiyor.")
            return


        fault_data = {
            "type": fault_type,
            "trigger_time": trigger_time_seconds,
            "conditions": conditions,
            "severity": severity_start,
            "progression_rate": progression_rate,
            "active": False, # Arıza henüz aktif değil, tetikleyici kontrol edecek
            "intermittent": intermittent,
            "intermittent_interval_s": intermittent_interval_s,
            "intermittent_duration_s": intermittent_duration_s,
            "last_intermittent_trigger": 0,
            "intermittent_active_until": 0,
            "details": details # Sensör arızaları için özel detaylar
        }
        self.fault_triggers.append(fault_data)
        print(f"Arıza tetikleyicisi eklendi: {fault_data['type']} (Zaman: {trigger_time_seconds}s, Koşullar: {conditions}, Aralıklı: {intermittent})")

    def _check_conditions(self, conditions, bus_state, current_sim_time_seconds):
        if not conditions:
            return False

        all_conditions_met = True
        for condition_key, threshold_value in conditions.items():
            parts = condition_key.split('_')
            sensor_name = '_'.join(parts[:-1])
            operator = parts[-1]

            actual_value = None
            if sensor_name == "sim_time":
                actual_value = current_sim_time_seconds
            elif sensor_name in bus_state:
                actual_value = bus_state[sensor_name]
            else:
                all_conditions_met = False
                break

            if operator == "gt":
                if not (actual_value > threshold_value): all_conditions_met = False; break
            elif operator == "lt":
                if not (actual_value < threshold_value): all_conditions_met = False; break
            elif operator == "eq":
                if not (actual_value == threshold_value): all_conditions_met = False; break
            elif operator == "gte":
                if not (actual_value >= threshold_value): all_conditions_met = False; break
            elif operator == "lte":
                if not (actual_value <= threshold_value): all_conditions_met = False; break
            else:
                all_conditions_met = False; break
        
        return all_conditions_met

    def apply_sensor_overrides(self, bus_state):
        for sensor_name, override_info in self.sensor_override_faults.items():
            if sensor_name in bus_state:
                if override_info["type"] == "frozen":
                    if "initial_value" not in override_info:
                        override_info["initial_value"] = bus_state[sensor_name]
                    bus_state[sensor_name] = override_info["initial_value"]
                elif override_info["type"] == "zero":
                    bus_state[sensor_name] = 0
                elif override_info["type"] == "offset":
                    bus_state[sensor_name] += override_info["value"]
                elif override_info["type"] == "noisy":
                    bus_state[sensor_name] = bus_state[sensor_name] + np.random.normal(0, DEFAULT_NOISE_STD_DEV.get(sensor_name, 0) + override_info["severity"] * 5)


    def update(self, current_sim_time_seconds, bus_state, dt):
        current_error_code = None
        current_health_status = "normal_calisma"

        for fault_trigger in self.fault_triggers:
            if not fault_trigger["active"]:
                triggered_by_time = fault_trigger["trigger_time"] is not None and current_sim_time_seconds >= fault_trigger["trigger_time"]
                triggered_by_conditions = fault_trigger["conditions"] is not None and self._check_conditions(fault_trigger["conditions"], bus_state, current_sim_time_seconds)

                if triggered_by_time or triggered_by_conditions:
                    fault_trigger["active"] = True
                    if not fault_trigger["intermittent"]:
                        self.triggered_fault_types.add(fault_trigger["type"])
                    print(f"--- ARIZA TETİKLENDİ: {fault_trigger['type']} --- (Süre: {round(current_sim_time_seconds)}s, Koşul: {fault_trigger['conditions'] if triggered_by_conditions else 'Zaman'}, Aralıklı: {fault_trigger['intermittent']})")
                    self.active_faults.append(fault_trigger)
                    if fault_trigger["intermittent"]:
                        fault_trigger["last_intermittent_trigger"] = current_sim_time_seconds
                        fault_trigger["intermittent_active_until"] = current_sim_time_seconds + fault["intermittent_duration_s"]
                    
                    if fault_trigger["type"].startswith("sensor_") and fault_trigger["details"]:
                        self.sensor_override_faults[fault_trigger["details"]["sensor"]] = {
                            "type": fault_trigger["type"].replace("sensor_", ""),
                            "value": fault_trigger["details"].get("offset"),
                            "severity": fault["severity"],
                            "initial_value": None
                        }

        faults_to_deactivate = []
        for fault in list(self.active_faults):
            is_currently_active = True
            if fault["intermittent"]:
                if current_sim_time_seconds < fault["intermittent_active_until"]:
                    pass
                elif current_sim_time_seconds >= fault["intermittent_active_until"] and \
                     current_sim_time_seconds < fault["last_intermittent_trigger"] + fault["intermittent_interval_s"]:
                    is_currently_active = False
                    if fault["type"].startswith("sensor_") and fault["details"] and fault["details"]["sensor"] in self.sensor_override_faults:
                        del self.sensor_override_faults[fault["details"]["sensor"]]
                        fault["details"]["initial_value"] = None
                elif current_sim_time_seconds >= fault["last_intermittent_trigger"] + fault["intermittent_interval_s"]:
                    is_currently_active = True
                    fault["last_intermittent_trigger"] = current_sim_time_seconds
                    fault["intermittent_active_until"] = current_sim_time_seconds + fault["intermittent_duration_s"]
                    print(f"--- Aralıklı Arıza Yeniden Aktif: {fault['type']} ---")
                    if fault["type"].startswith("sensor_") and fault["details"]:
                        self.sensor_override_faults[fault["details"]["sensor"]] = {
                            "type": fault["type"].replace("sensor_", ""),
                            "value": fault["details"].get("offset"),
                            "severity": fault["severity"],
                            "initial_value": None
                        }
                else:
                    is_currently_active = False

            if not is_currently_active:
                continue

            if not fault["intermittent"]:
                fault["severity"] = min(1.0, fault["severity"] + fault["progression_rate"] * dt)
            elif fault["intermittent"] and fault["type"].startswith("sensor_noisy"):
                fault["severity"] = min(1.0, fault["severity"] + fault["progression_rate"] * dt * 5)


            if fault["type"] == "battery_overheat":
                current_health_status = "batarya_sicaklik_problemi"
                current_error_code = "BMS-T001"
                
                bus_state["batteryTempMin"] += fault["severity"] * 0.5 * dt * random.uniform(0.8, 1.2)
                bus_state["batteryTempMax"] += fault["severity"] * 0.8 * dt * random.uniform(0.8, 1.2)
                bus_state["batteryHealth"] = max(0, bus_state["batteryHealth"] - fault["severity"] * 0.0005 * dt)
                
                if bus_state["batteryTempMax"] > 65:
                     bus_state["batteryCurrent"] += abs(bus_state["batteryCurrent"]) * fault["severity"] * 0.05

            elif fault["type"] == "motor_insulation_degradation":
                current_health_status = "motor_izolasyon_problemi"
                current_error_code = "MOT-I002"
                
                bus_state["motorCurrent"] *= (1 + fault["severity"] * 0.05)
                bus_state["motorTemperature"] += fault["severity"] * 1.0 * dt * random.uniform(0.9, 1.1)
                bus_state["motorRPM"] += random.uniform(-100, 100) * fault["severity"]
                bus_state["motorRPM"] = max(0, bus_state["motorRPM"])

            elif fault["type"] == "tire_pressure_loss":
                current_health_status = "lastik_basinci_dusuk"
                current_error_code = "TIRE-P001"
                
                bus_state["tirePressure"] = max(10, 80 - fault["severity"] * 70)
                
                bus_state["batteryCurrent"] += abs(bus_state["batteryCurrent"]) * fault["severity"] * 0.02
                bus_state["vehicleSpeed"] += random.uniform(-0.5, 0.5) * fault["severity"]

            elif fault["type"] == "coolant_pump_failure":
                current_health_status = "sogutma_pompasi_arizasi"
                current_error_code = "COOL-P001"
                
                bus_state["motorTemperature"] += fault["severity"] * 5.0 * dt * random.uniform(0.8, 1.2)
                bus_state["batteryTempMax"] += fault["severity"] * 3.0 * dt * random.uniform(0.8, 1.2)
                bus_state["batteryTempMin"] += fault["severity"] * 3.0 * dt * random.uniform(0.8, 1.2)
                bus_state["coolantTemp"] = bus_state["motorTemperature"]
                
                if bus_state["motorTemperature"] > 140 or bus_state["batteryTempMax"] > 65:
                    bus_state["motorCurrent"] *= (1 - fault["severity"] * 0.3)

            elif fault["type"] == "aux_battery_degradation":
                current_health_status = "yardimci_aku_performans_dusuk"
                current_error_code = "AUX-B001"
                
                bus_state["auxBatteryVoltage"] = max(18.0, 24.0 - fault["severity"] * 6.0)

            elif fault["type"] == "sensor_frozen":
                current_health_status = f"{fault['details']['sensor']}_sensor_frozen"
                current_error_code = "SENSOR-F001"

            elif fault["type"] == "sensor_offset":
                current_health_status = f"{fault['details']['sensor']}_sensor_offset"
                current_error_code = "SENSOR-O002"

            elif fault["type"] == "sensor_noisy":
                current_health_status = f"{fault['details']['sensor']}_sensor_noisy"
                current_error_code = "SENSOR-N003"
            
        self.apply_sensor_overrides(bus_state)

        return current_health_status, current_error_code

class RouteManager:
    def __init__(self, route_data):
        self.route_data = route_data
        self.current_segment_index = 0
        self.distance_in_current_segment_km = 0.0
        
        self.current_latitude = 37.0000
        self.current_longitude = 35.3250
        self.current_bearing_degrees = 330

        self.current_slope_degrees = 0.0
        self.current_speed_limit_kph = 0
        self.current_traffic_density = "low"
        self.current_action = "drive"
        
        if self.route_data:
            self._update_current_segment_info()

    def _update_current_segment_info(self):
        if self.current_segment_index < len(self.route_data):
            segment = self.route_data[self.current_segment_index]
            self.current_slope_degrees = segment.get("slope_degrees", 0.0)
            self.current_speed_limit_kph = segment.get("speed_limit_kph", 0)
            self.current_traffic_density = segment.get("traffic_density", "low")
            self.current_action = segment.get("action", "drive")
            if "initial_bearing" in segment:
                self.current_bearing_degrees = segment["initial_bearing"]
            
        else:
            self.current_slope_degrees = 0.0
            self.current_speed_limit_kph = 0
            self.current_traffic_density = "none"
            self.current_action = "end_of_route"
            
    def update(self, distance_traveled_dt_km):
        if distance_traveled_dt_km > 0:
            self.current_latitude, self.current_longitude = calculate_new_coords(
                self.current_latitude, self.current_longitude,
                self.current_bearing_degrees, distance_traveled_dt_km
            )

        self.distance_in_current_segment_km += distance_traveled_dt_km

        if self.current_segment_index < len(self.route_data):
            current_segment = self.route_data[self.current_segment_index]
            if self.distance_in_current_segment_km >= current_segment["distance_km"]:
                self.distance_in_current_segment_km -= current_segment["distance_km"]
                self.current_segment_index += 1
                self._update_current_segment_info()
                print(f"--- Rota Segmenti Değişti --- Yeni Segment: Index {self.current_segment_index}, Eğim: {self.current_slope_degrees}°,"
                      f" Hız Limiti: {self.current_speed_limit_kph} km/s, Aksiyon: {self.current_action}, Yön: {self.current_bearing_degrees}°")
        
        return self.get_state()

    def get_state(self):
        return {
            "current_slope_degrees": round(apply_noise(self.current_slope_degrees, "current_slope_degrees"), 2),
            "current_speed_limit_kph": self.current_speed_limit_kph,
            "current_traffic_density": self.current_traffic_density,
            "current_route_action": self.current_action,
            "current_segment_index": self.current_segment_index,
            "distance_in_current_segment_km": round(self.distance_in_current_segment_km, 2),
            "latitude": round(apply_noise(self.current_latitude, "latitude"), 6),
            "longitude": round(apply_noise(self.current_longitude, "longitude"), 6),
            "bearing_degrees": round(apply_noise(self.current_bearing_degrees, "bearing_degrees"), 1)
        }