import random

def generate_bodies(filename, num_bodies, mass_range, pos_range, vel_range):
    """
    Generuje plik z losowymi wartościami w formacie:
    masa, x, y, z, vx, vy, vz

    Args:
        filename (str): Nazwa pliku wyjściowego.
        num_bodies (int): Liczba ciał do wygenerowania.
        mass_range (tuple): Zakres mas (min, max).
        pos_range (tuple): Zakres współrzędnych pozycji (min, max).
        vel_range (tuple): Zakres współrzędnych prędkości (min, max).
    """
    with open(filename, 'w') as f:
        for _ in range(num_bodies):
            mass = random.uniform(*mass_range)
            x, y, z = (random.uniform(*pos_range) for _ in range(3))
            vx, vy, vz = (random.uniform(*vel_range) for _ in range(3))
            f.write(f"{mass:.6e} {x:.3f} {y:.3f} {z:.3f} {vx:.3f} {vy:.3f} {vz:.3f}\n")

# Przykładowe użycie
generate_bodies(
    filename="random.txt",
    num_bodies=512,
    mass_range=(1e20, 1e30),  # Zakres mas
    pos_range=(-1e3, 1e3),    # Zakres pozycji
    vel_range=(-1e1, 1e1)     # Zakres prędkości
)