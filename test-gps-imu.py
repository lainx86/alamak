from pymavlink import mavutil
import time
import math
import pygame


PORT = "/dev/ttyACM0"
BAUD = 115200

WIDTH = 1100
HEIGHT = 700
FPS = 30


def deg_from_e7(value):
    return value / 1e7


def m_from_mm(value):
    return value / 1000.0


def rad_s_to_deg_s(value):
    return value * 180.0 / math.pi


def connect_pixhawk(port, baud):
    print(f"[INIT] Menghubungkan ke {port} dengan baudrate {baud}...")
    master = mavutil.mavlink_connection(port, baud=baud)

    print("[INIT] Menunggu heartbeat dari Pixhawk...")
    master.wait_heartbeat()

    print("[OK] Heartbeat diterima")
    print(f"[INFO] target_system    = {master.target_system}")
    print(f"[INFO] target_component = {master.target_component}")

    return master


def request_message_interval(master, message_id, frequency_hz):
    interval_us = int(1_000_000 / frequency_hz)

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        interval_us,
        0,
        0,
        0,
        0,
        0
    )


def setup_streams(master):
    request_message_interval(
        master,
        mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
        5
    )

    request_message_interval(
        master,
        mavutil.mavlink.MAVLINK_MSG_ID_SCALED_IMU,
        20
    )

    request_message_interval(
        master,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        20
    )

    print("[OK] Request stream GPS dan IMU dikirim")


def draw_text(screen, font, text, x, y, color=(255, 255, 255)):
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))


def draw_artificial_horizon(screen, font, roll_deg, pitch_deg):
    cx = 320
    cy = 300
    radius = 180

    pygame.draw.circle(screen, (230, 230, 230), (cx, cy), radius + 4, 3)

    horizon_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    hs_cx = radius
    hs_cy = radius

    pitch_offset = pitch_deg * 3.0

    pygame.draw.rect(
        horizon_surface,
        (60, 130, 210),
        (0, -radius * 2 + hs_cy + pitch_offset, radius * 2, radius * 2)
    )

    pygame.draw.rect(
        horizon_surface,
        (120, 80, 40),
        (0, hs_cy + pitch_offset, radius * 2, radius * 2)
    )

    pygame.draw.line(
        horizon_surface,
        (255, 255, 255),
        (0, hs_cy + pitch_offset),
        (radius * 2, hs_cy + pitch_offset),
        4
    )

    rotated = pygame.transform.rotate(horizon_surface, roll_deg)
    rect = rotated.get_rect(center=(cx, cy))

    mask_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (radius, radius), radius)

    clipped = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    clipped.blit(horizon_surface, (0, 0))
    clipped.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    rotated_clipped = pygame.transform.rotate(clipped, roll_deg)
    rotated_rect = rotated_clipped.get_rect(center=(cx, cy))
    screen.blit(rotated_clipped, rotated_rect)

    pygame.draw.circle(screen, (230, 230, 230), (cx, cy), radius, 3)

    pygame.draw.line(screen, (255, 255, 0), (cx - 60, cy), (cx - 15, cy), 4)
    pygame.draw.line(screen, (255, 255, 0), (cx + 15, cy), (cx + 60, cy), 4)
    pygame.draw.circle(screen, (255, 255, 0), (cx, cy), 5)

    pygame.draw.polygon(
        screen,
        (255, 255, 0),
        [
            (cx, cy - radius - 15),
            (cx - 10, cy - radius + 5),
            (cx + 10, cy - radius + 5)
        ]
    )

    draw_text(screen, font, "ARTIFICIAL HORIZON", cx - 110, cy - radius - 55)
    draw_text(screen, font, f"Roll  : {roll_deg:7.2f} deg", cx - 90, cy + radius + 25)
    draw_text(screen, font, f"Pitch : {pitch_deg:7.2f} deg", cx - 90, cy + radius + 50)


def draw_yaw_indicator(screen, font, yaw_deg):
    cx = 770
    cy = 180
    radius = 95

    pygame.draw.circle(screen, (230, 230, 230), (cx, cy), radius, 3)

    for angle in range(0, 360, 30):
        rad = math.radians(angle - 90)
        x1 = cx + math.cos(rad) * (radius - 12)
        y1 = cy + math.sin(rad) * (radius - 12)
        x2 = cx + math.cos(rad) * radius
        y2 = cy + math.sin(rad) * radius
        pygame.draw.line(screen, (230, 230, 230), (x1, y1), (x2, y2), 2)

    labels = {
        0: "N",
        90: "E",
        180: "S",
        270: "W"
    }

    for angle, label in labels.items():
        rad = math.radians(angle - 90)
        x = cx + math.cos(rad) * (radius - 28)
        y = cy + math.sin(rad) * (radius - 28)
        draw_text(screen, font, label, x - 7, y - 10)

    yaw_rad = math.radians(yaw_deg - 90)
    x = cx + math.cos(yaw_rad) * (radius - 20)
    y = cy + math.sin(yaw_rad) * (radius - 20)

    pygame.draw.line(screen, (255, 80, 80), (cx, cy), (x, y), 5)
    pygame.draw.circle(screen, (255, 80, 80), (cx, cy), 6)

    draw_text(screen, font, "YAW / HEADING", cx - 65, cy - radius - 35)
    draw_text(screen, font, f"{yaw_deg:7.2f} deg", cx - 45, cy + radius + 20)


def draw_bar(screen, font, label, value, unit, x, y, max_abs, color):
    bar_width = 280
    bar_height = 22
    center_x = x + bar_width // 2

    pygame.draw.rect(screen, (80, 80, 80), (x, y, bar_width, bar_height), 2)
    pygame.draw.line(screen, (200, 200, 200), (center_x, y - 3), (center_x, y + bar_height + 3), 2)

    clipped = max(-max_abs, min(max_abs, value))
    length = int((clipped / max_abs) * (bar_width / 2))

    if length >= 0:
        rect = (center_x, y + 3, length, bar_height - 6)
    else:
        rect = (center_x + length, y + 3, -length, bar_height - 6)

    pygame.draw.rect(screen, color, rect)

    draw_text(screen, font, f"{label}: {value:8.2f} {unit}", x, y - 24)


def draw_imu_bars(screen, font, latest_imu, latest_attitude):
    x = 650
    y = 360

    draw_text(screen, font, "IMU SENSOR BARS", x, y - 45)

    if latest_imu is None:
        draw_text(screen, font, "Belum ada data IMU", x, y)
        return

    draw_bar(screen, font, "Accel X", latest_imu["accel_x_mg"], "mg", x, y, 2000, (80, 180, 255))
    draw_bar(screen, font, "Accel Y", latest_imu["accel_y_mg"], "mg", x, y + 65, 2000, (80, 180, 255))
    draw_bar(screen, font, "Accel Z", latest_imu["accel_z_mg"], "mg", x, y + 130, 2000, (80, 180, 255))

    draw_bar(screen, font, "Gyro X", latest_imu["gyro_x_mrad_s"], "mrad/s", x, y + 220, 5000, (255, 170, 70))
    draw_bar(screen, font, "Gyro Y", latest_imu["gyro_y_mrad_s"], "mrad/s", x, y + 285, 5000, (255, 170, 70))
    draw_bar(screen, font, "Gyro Z", latest_imu["gyro_z_mrad_s"], "mrad/s", x, y + 350, 5000, (255, 170, 70))

    if latest_attitude is not None:
        draw_text(screen, font, f"Roll Speed : {latest_attitude['rollspeed_deg_s']:8.2f} deg/s", x, y + 435)
        draw_text(screen, font, f"Pitch Speed: {latest_attitude['pitchspeed_deg_s']:8.2f} deg/s", x, y + 460)
        draw_text(screen, font, f"Yaw Speed  : {latest_attitude['yawspeed_deg_s']:8.2f} deg/s", x, y + 485)


def draw_gps_panel(screen, font, latest_gps):
    x = 30
    y = 30

    draw_text(screen, font, "GPS DATA", x, y)

    if latest_gps is None:
        draw_text(screen, font, "Belum ada data GPS", x, y + 30)
        return

    lines = [
        f"Fix Type          : {latest_gps['fix_type']}",
        f"Latitude          : {latest_gps['lat']:.7f}",
        f"Longitude         : {latest_gps['lon']:.7f}",
        f"Altitude          : {latest_gps['alt_m']:.2f} m",
        f"Ground Speed      : {latest_gps['vel_m_s']:.2f} m/s",
        f"Course Over Ground: {latest_gps['cog_deg']:.2f} deg",
        f"Satelit Terlihat  : {latest_gps['satellites_visible']}",
        f"EPH               : {latest_gps['eph']}",
        f"EPV               : {latest_gps['epv']}"
    ]

    for i, line in enumerate(lines):
        draw_text(screen, font, line, x, y + 30 + i * 22)


def main():
    master = connect_pixhawk(PORT, BAUD)
    setup_streams(master)

    latest_gps = None
    latest_imu = None
    latest_attitude = None

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pixhawk 6C GPS dan IMU Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    running = True

    print("[INFO] Visualisasi GPS dan IMU aktif")
    print("[INFO] Tutup window atau tekan Ctrl+C untuk berhenti")

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            while True:
                msg = master.recv_match(
                    type=[
                        "GPS_RAW_INT",
                        "SCALED_IMU",
                        "ATTITUDE"
                    ],
                    blocking=False
                )

                if msg is None:
                    break

                msg_type = msg.get_type()

                if msg_type == "GPS_RAW_INT":
                    latest_gps = {
                        "fix_type": msg.fix_type,
                        "lat": deg_from_e7(msg.lat),
                        "lon": deg_from_e7(msg.lon),
                        "alt_m": m_from_mm(msg.alt),
                        "eph": msg.eph,
                        "epv": msg.epv,
                        "vel_m_s": msg.vel / 100.0,
                        "cog_deg": msg.cog / 100.0,
                        "satellites_visible": msg.satellites_visible
                    }

                elif msg_type == "SCALED_IMU":
                    latest_imu = {
                        "accel_x_mg": msg.xacc,
                        "accel_y_mg": msg.yacc,
                        "accel_z_mg": msg.zacc,
                        "gyro_x_mrad_s": msg.xgyro,
                        "gyro_y_mrad_s": msg.ygyro,
                        "gyro_z_mrad_s": msg.zgyro,
                        "mag_x_mgauss": msg.xmag,
                        "mag_y_mgauss": msg.ymag,
                        "mag_z_mgauss": msg.zmag
                    }

                elif msg_type == "ATTITUDE":
                    latest_attitude = {
                        "roll_deg": math.degrees(msg.roll),
                        "pitch_deg": math.degrees(msg.pitch),
                        "yaw_deg": math.degrees(msg.yaw) % 360.0,
                        "rollspeed_deg_s": rad_s_to_deg_s(msg.rollspeed),
                        "pitchspeed_deg_s": rad_s_to_deg_s(msg.pitchspeed),
                        "yawspeed_deg_s": rad_s_to_deg_s(msg.yawspeed)
                    }

            screen.fill((20, 20, 25))

            draw_gps_panel(screen, font, latest_gps)

            if latest_attitude is None:
                draw_text(screen, font, "Belum ada data attitude", 180, 300)
                draw_yaw_indicator(screen, font, 0.0)
            else:
                draw_artificial_horizon(
                    screen,
                    font,
                    latest_attitude["roll_deg"],
                    latest_attitude["pitch_deg"]
                )

                draw_yaw_indicator(
                    screen,
                    font,
                    latest_attitude["yaw_deg"]
                )

            draw_imu_bars(screen, font, latest_imu, latest_attitude)

            pygame.display.flip()
            clock.tick(FPS)

    except KeyboardInterrupt:
        print("\n[STOP] Program dihentikan oleh user")

    pygame.quit()


if __name__ == "__main__":
    main()