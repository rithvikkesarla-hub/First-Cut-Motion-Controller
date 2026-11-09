
import cv2
import mediapipe as mp
import requests
import time
import math
import threading
import sys

from pynput.keyboard import Controller, Key
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# CONFIGURATION
# =========================================================

BASE_URL = "" #Add phyphox ka IP here

MODEL_PATH = "pose_landmarker_lite.task"

POLL_INTERVAL = 0.05

ATTACK_THRESHOLD = 12.0

GRAVITY_Z_MIN = 8.0
GRAVITY_Z_MAX = 10.0

DIRECTION_CONFIRMATION = 3
GUARD_RELEASE_DELAY = 0.1

MOVEMENT_THRESHOLD = 0.12
SMOOTHING = 0.35

# DUCK CONFIGURATION
DUCK_THRESHOLD = 0.5595
DUCK_KEY = 'x'
DUCK_TAP_DURATION = 0.05


# =========================================================
# SHARED KEYBOARD
# =========================================================

keyboard = Controller()
keyboard_lock = threading.Lock()

stop_event = threading.Event()

held_guard_key = None
held_movement_key = None

duck_latched = False


def press_key(key):
    with keyboard_lock:
        keyboard.press(key)


def release_key(key):
    with keyboard_lock:
        keyboard.release(key)


def tap_key(key):
    with keyboard_lock:
        keyboard.press(key)
        time.sleep(DUCK_TAP_DURATION)
        keyboard.release(key)


# =========================================================
# CV MOVEMENT CONTROL
# CV OWNS ONLY LEFT AND RIGHT
# =========================================================

def update_movement(direction):
    global held_movement_key

    if direction == "LEFT":
        new_key = Key.left
    elif direction == "RIGHT":
        new_key = Key.right
    else:
        new_key = None

    if new_key == held_movement_key:
        return

    with keyboard_lock:
        if held_movement_key is not None:
            keyboard.release(held_movement_key)

        held_movement_key = new_key

        if held_movement_key is not None:
            keyboard.press(held_movement_key)


def movement_camera():
    global held_movement_key
    global duck_latched

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    BaseOptions = python.BaseOptions

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    connections = [
        (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28)
    ]

    previous_center_x = None
    previous_time = None
    smoothed_velocity = 0.0
    last_timestamp = 0

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:

            while not stop_event.is_set():

                success, frame = cap.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                current_time = time.monotonic()

                timestamp_ms = int(current_time * 1000)
                timestamp_ms = max(timestamp_ms, last_timestamp + 1)
                last_timestamp = timestamp_ms

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                movement = "STILL"
                duck_status = "NOT DUCKING"

                if result.pose_landmarks:

                    landmarks = result.pose_landmarks[0]

                    # -----------------------------------------
                    # DUCK DETECTION
                    # -----------------------------------------

                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]

                    if (
                        left_shoulder.visibility > 0.5
                        and right_shoulder.visibility > 0.5
                    ):

                        shoulder_y = (
                            left_shoulder.y + right_shoulder.y
                        ) / 2

                        if shoulder_y > DUCK_THRESHOLD:

                            duck_status = "DUCKING"

                            if not duck_latched:
                                tap_key(DUCK_KEY)
                                duck_latched = True
                                print("\nDUCK DETECTED - X TAPPED")

                        else:
                            duck_latched = False

                    # -----------------------------------------
                    # DRAW SKELETON
                    # -----------------------------------------

                    for start, end in connections:

                        x1 = int(landmarks[start].x * frame.shape[1])
                        y1 = int(landmarks[start].y * frame.shape[0])

                        x2 = int(landmarks[end].x * frame.shape[1])
                        y2 = int(landmarks[end].y * frame.shape[0])

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                    # Draw landmarks
                    for landmark in landmarks:

                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 0, 255),
                            -1
                        )

                    # -----------------------------------------
                    # HIP CENTER MOVEMENT DETECTION
                    # -----------------------------------------

                    center_x = (
                        landmarks[23].x + landmarks[24].x
                    ) / 2

                    if previous_center_x is not None:

                        dt = current_time - previous_time

                        if dt > 0:

                            velocity = (
                                center_x - previous_center_x
                            ) / dt

                            smoothed_velocity = (
                                SMOOTHING * velocity
                                + (1 - SMOOTHING) * smoothed_velocity
                            )

                            if smoothed_velocity < -MOVEMENT_THRESHOLD:
                                movement = "LEFT"

                            elif smoothed_velocity > MOVEMENT_THRESHOLD:
                                movement = "RIGHT"

                    previous_center_x = center_x
                    previous_time = current_time

                else:

                    previous_center_x = None
                    previous_time = None
                    smoothed_velocity = 0.0

                # CV updates only left/right keys.
                update_movement(movement)

                # -----------------------------------------
                # DISPLAY STATUS
                # -----------------------------------------

                cv2.putText(
                    frame,
                    f"MOVEMENT: {movement}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"DUCK: {duck_status}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.imshow("Movement Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

    finally:

        update_movement("STILL")

        cap.release()
        cv2.destroyAllWindows()


# =========================================================
# IMU SWORD CONTROL
# IMU OWNS ONLY UP, DOWN AND Z
# =========================================================

attack_active = False
attack_type = None

guard_release_pending = False
guard_release_time = 0

previous_gravity_y = None
direction_trend = None
direction_count = 0

current_guard_position = "NORMAL"


def set_guard_key(new_key):
    global held_guard_key

    if held_guard_key == new_key:
        return

    with keyboard_lock:

        if held_guard_key is not None:
            keyboard.release(held_guard_key)

        held_guard_key = new_key

        if held_guard_key is not None:
            keyboard.press(held_guard_key)


def update_guard(position):
    global current_guard_position

    current_guard_position = position

    if attack_active or guard_release_pending:
        return

    if position == "UP":
        set_guard_key(Key.up)

    elif position == "DOWN":
        set_guard_key(Key.down)

    else:
        set_guard_key(None)


# =========================================================
# ATTACK CONTROL
# Z PRESSED BEFORE GUARD TRANSITION
# =========================================================

def start_attack(direction=None, attack_kind="DIRECTIONAL"):
    global attack_active, attack_type, held_guard_key

    if attack_active:
        return

    attack_active = True
    attack_type = attack_kind

    # Decide required guard key.
    if attack_kind == "HORIZONTAL":
        new_guard_key = None

    elif direction == "UP TO DOWN":
        new_guard_key = Key.up

    elif direction == "DOWN TO UP":
        new_guard_key = Key.down

    else:
        new_guard_key = None

    # Press Z immediately after recognition.
    with keyboard_lock:

        keyboard.press('z')

        # Change guard after Z has been pressed.
        if held_guard_key != new_guard_key:

            if held_guard_key is not None:
                keyboard.release(held_guard_key)

            held_guard_key = new_guard_key

            if held_guard_key is not None:
                keyboard.press(held_guard_key)

    print(f"\n{attack_kind} ATTACK TRIGGERED")


def release_attack():
    global attack_active, attack_type
    global guard_release_pending, guard_release_time

    if not attack_active:
        return

    # Release Z first.
    release_key('z')

    attack_active = False
    attack_type = None

    # Then wait before releasing the guard key.
    guard_release_pending = True
    guard_release_time = time.time() + GUARD_RELEASE_DELAY

    print("\nATTACK RELEASED")


def finish_guard_release():
    global guard_release_pending

    if guard_release_pending and time.time() >= guard_release_time:

        set_guard_key(None)

        guard_release_pending = False


def force_horizontal_mode():
    global attack_active, attack_type
    global guard_release_pending
    global direction_trend, direction_count

    if attack_active and attack_type == "DIRECTIONAL":

        release_key('z')

        attack_active = False
        attack_type = None

    set_guard_key(None)

    # Ensure no directional guard key remains held.
    release_key(Key.up)
    release_key(Key.down)

    guard_release_pending = False
    direction_trend = None
    direction_count = 0


# =========================================================
# PHYPhox DATA
# =========================================================

def get_sensor_data():

    response = requests.get(
        f"{BASE_URL}/get?linX&linY&linZ&graY&graZ",
        timeout=2
    )

    response.raise_for_status()

    data = response.json()["buffer"]

    lin_x = data["linX"]["buffer"][-1]
    lin_y = data["linY"]["buffer"][-1]
    lin_z = data["linZ"]["buffer"][-1]

    gravity_y = data["graY"]["buffer"][-1]
    gravity_z = data["graZ"]["buffer"][-1]

    return lin_x, lin_y, lin_z, gravity_y, gravity_z


# =========================================================
# MAIN IMU LOOP
# Webcam runs independently in another thread.
# =========================================================

camera_thread = threading.Thread(
    target=movement_camera,
    daemon=True
)

try:

    print("Connecting to phyphox...")

    requests.get(
        f"{BASE_URL}/control?cmd=clear",
        timeout=2
    )

    requests.get(
        f"{BASE_URL}/control?cmd=start",
        timeout=2
    )

    print("Controller started!")
    print("Switch to First Cut.")
    print("--------------------------------")

    camera_thread.start()

    while not stop_event.is_set():

        try:

            # IMU polling
            lin_x, lin_y, lin_z, gravity_y, gravity_z = get_sensor_data()

            acceleration = math.sqrt(
                lin_x ** 2 + lin_y ** 2 + lin_z ** 2
            )

            # ---------------------------------------------
            # HORIZONTAL ATTACK HAS PRIORITY
            # ---------------------------------------------

            if GRAVITY_Z_MIN < gravity_z < GRAVITY_Z_MAX:

                force_horizontal_mode()

                if acceleration > ATTACK_THRESHOLD:

                    if not attack_active:

                        start_attack(
                            direction=None,
                            attack_kind="HORIZONTAL"
                        )

                else:

                    if attack_active and attack_type == "HORIZONTAL":
                        release_attack()

                guard_position = "IGNORED"
                detection_mode = "HORIZONTAL"

            # ---------------------------------------------
            # NORMAL DIRECTIONAL ATTACK AND GUARD
            # ---------------------------------------------

            else:

                finish_guard_release()

                if gravity_y < 2.0:
                    guard_position = "DOWN"

                elif gravity_y > 6.0:
                    guard_position = "UP"

                else:
                    guard_position = "NORMAL"

                current_guard_position = guard_position

                if attack_active and acceleration <= ATTACK_THRESHOLD:
                    release_attack()

                if not attack_active and not guard_release_pending:

                    if acceleration > ATTACK_THRESHOLD:

                        if previous_gravity_y is not None:

                            if gravity_y < previous_gravity_y:
                                new_trend = "UP TO DOWN"

                            elif gravity_y > previous_gravity_y:
                                new_trend = "DOWN TO UP"

                            else:
                                new_trend = None

                            if new_trend is not None:

                                if new_trend == direction_trend:
                                    direction_count += 1

                                else:
                                    direction_trend = new_trend
                                    direction_count = 1

                                if direction_count >= DIRECTION_CONFIRMATION:

                                    start_attack(
                                        direction=direction_trend,
                                        attack_kind="DIRECTIONAL"
                                    )

                                    direction_count = 0

                    else:

                        direction_trend = None
                        direction_count = 0

                if not attack_active and not guard_release_pending:
                    update_guard(guard_position)

                detection_mode = "DIRECTIONAL"

            previous_gravity_y = gravity_y

            status = (
                f"ACC: {acceleration:6.2f} | "
                f"GY: {gravity_y:6.2f} | "
                f"GZ: {gravity_z:6.2f} | "
                f"GUARD: {guard_position:8} | "
                f"MODE: {detection_mode:12} | "
                f"ATTACK: {attack_type or 'NONE':12}"
            )

            sys.stdout.write("\r\033[K" + status)
            sys.stdout.flush()

            time.sleep(POLL_INTERVAL)

        except (
            requests.RequestException,
            KeyError,
            IndexError,
            ValueError
        ) as error:

            print(f"\nSensor read error: {error}")
            time.sleep(0.2)


except KeyboardInterrupt:

    print("\n\nController stopped by user.")


finally:

    stop_event.set()

    # Release IMU-owned keys.
    release_key('z')
    set_guard_key(None)

    # Stop camera thread.
    if camera_thread.is_alive():
        camera_thread.join(timeout=3)

    requests.get(
        f"{BASE_URL}/control?cmd=stop",
        timeout=2
    )

    print("\nAll keys released. phyphox stopped.")
