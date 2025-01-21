import numpy as np

# Stała grawitacji
G = 6.67430e-11  # m^3 kg^-1 s^-2

# Parametry ciała centralnego
central_mass = 1e30  # Masa centralnego ciała (np. gwiazda)
central_radius = 1e9  # Promień centralnego ciała

# Parametry pierścienia
n_particles = 50  # Liczba obiektów w pierścieniu
particle_mass = 1e20  # Masa każdego obiektu w pierścieniu
particle_radius = 1e4  # Promień każdego obiektu
radius_mean = 1.1e11  # Średnia odległość pierścienia od centralnego ciała
radius_variation = 1e9  # Maksymalne odchylenie od średniego promienia
z_variation = 1e9  # Zakres losowej pozycji w osi Z
inclination_variation = 5  # Maksymalne nachylenie orbity (w stopniach)
speed_variation = 1000.0  # Maksymalne zaburzenie prędkości orbitalnej

# Tworzenie warunków początkowych
with open("initial_conditions.txt", "w") as file:
    # Centralne ciało
    file.write(f"{central_mass} {central_radius} 0.0 0.0 0.0 0.0 0.0 0.0\n")

    # Cząsteczki pierścienia
    for i in range(n_particles):
        # Losowy promień orbity
        r = radius_mean + np.random.uniform(-radius_variation, radius_variation)

        # Wyliczanie prędkości orbitalnej na podstawie prawa grawitacji
        v_orbital = np.sqrt(G * central_mass / r)

        # Losowy kąt pozycji w płaszczyźnie pierścienia
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Losowe odchylenie w osi Z
        z = np.random.uniform(-z_variation, z_variation)

        # Zaburzenie inklinacji orbity (nachylenie względem płaszczyzny XY)
        inclination = np.radians(
            np.random.uniform(-inclination_variation, inclination_variation)
        )
        v_orbital_z = v_orbital * np.sin(inclination)  # Prędkość w osi Z
        v_orbital_xy = v_orbital * np.cos(inclination)  # Prędkość w płaszczyźnie XY

        # Prędkość orbitalna w kierunku stycznym
        vx = -v_orbital_xy * np.sin(angle) + np.random.uniform(
            -speed_variation, speed_variation
        )
        vy = v_orbital_xy * np.cos(angle) + np.random.uniform(
            -speed_variation, speed_variation
        )
        vz = v_orbital_z + np.random.uniform(
            -speed_variation / 10, speed_variation / 10
        )

        # Zapis do pliku
        file.write(f"{particle_mass} {particle_radius} {x} {y} {z} {vx} {vy} {vz}\n")
