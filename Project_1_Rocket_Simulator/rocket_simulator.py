import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class RocketSimulator:
    def __init__(self, rocket_name="RFA Technologie Demonstrator"):
        #Rocket properties
        self.rocket_name = rocket_name
        self.m_empty = 1000           # kg (empty mass)
        self.m_fuel = 5000            # kg (fuel mass)
        self.m_payload = 500          # kg (payload mass)
        self.thrust = 70000           # N (Newtons)
        self.burn_time = 180          # seconds
        self.isp = 280                # specific impulse (seconds)
        self.cd = 0.3                 # drag coefficient
        self.area = 2.5               # cross-sectional area (m^2)
        self.g0 = 9.81
        # sets up the parameters of the rocket (using Falcon 1 class values)

    def atmosphere_model (self, altitude_km):
        """
        Return atmospheric density at given altitude (km)
        Uses exponential atmosphere model
        """
        #print(f"DEBUG atmosphere_model: altitude_km={altitude_km}, H=8.5")
        rho0 = 1.225                  # density at sea level (kg/m^3)
        H = 8.5                      # scale height (kmeters)
        result = rho0 * np.exp(-altitude_km / H)
        #print(f"DEBUG atmosphere_model: returning rho={result}")
        return result
        #this calculates air density at any altitude

    def rocket_mass(self, t):
        """Current rocket mass at time t"""
        if t < self.burn_time:
            #Mass flow rate (simplified model)
            fuel_fraction = (self.burn_time - t) / self.burn_time
            return self.m_empty + self.m_fuel * fuel_fraction + self.m_payload 
        else:
            return self.m_empty + self.m_payload
        #this calculates mass as fuel burns out
    
    def rocket_thrust(self, t):
        """Thrust profile over time"""
        if t < self.burn_time:
            return self.thrust
        else:
            return 0
        # rocket thrust should be constant until burnout

    def rocket_dynamics(self, state, t):
        """
        State = [altitude, velocity]
        Returns d(state)/dt
        """
        altitude_m, velocity = state

        # Get current properties
        altitude_km = altitude_m/1000.0

        mass = self.rocket_mass(t)
        thrust = self.rocket_thrust(t)
        
        # Physics calculations
        g = 9.81 * (6371 / (6371 + altitude_km))**2 # gravity varies with altitude
        rho = self.atmosphere_model(altitude_km)

        # Drag force
        drag = 0.5 * rho * velocity**2 * self.cd * self.area

        # Accelerations
        dv_dt = (thrust - drag) / mass - g
        dy_dt = velocity

        return [dy_dt, dv_dt]
        #this is the physics engine: converts state

    def simulate(self, t_max=600, dt=0.1):
        """
        Simulate rocket flight
        Returns: time, altitude, velocity, acceleration arrays
        """
        t = np.arange(0, t_max, dt)

        #Initial state: ground level, zero velocity
        state0 = [0.0, 0.0]        # [altitude km, velocity m/s]

        # Solve ODE
        trajectory = odeint(self.rocket_dynamics, state0, t)

        altitude_m = trajectory[:,0]
        velocity = trajectory[:,1]
        altitude_km = altitude_m/1000.0

        apogee_idx = np.argmax(altitude_km)

        landing_indices = np.where(altitude_km[apogee_idx:] < 0.5)[0]
        if len(landing_indices) > 0:
            cutoff = landing_indices[0]
            t=t[:cutoff]
            altitude_km = altitude_km[:cutoff]
            velocity = velocity[:cutoff]
            trajectory = trajectory[:cutoff]
    
        #Calculation acceleration
        acceleration = np.zeros(len(t))
        for i in range(len(t)):
            acc = self.rocket_dynamics(trajectory[i], t[i])[1]
            acceleration[i] = acc

        return t, altitude_km, velocity, acceleration
        #this runs the simulation and returns results

    def plot_trajectory(self, t, alt, vel, acc):
        """Plot trajectoary results"""
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        #Altitude vs time
        axes[0].plot(t, alt, 'b-', linewidth=2, label='Altitude')
        axes[0].set_ylabel('Altitude (km)', fontsize=11)
        axes[0].set_title('Rocket Ascent Trajectory', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        #Velocity vs time
        axes[1].plot(t, vel, 'g-', linewidth=2, label='Velocity')
        axes[1].set_ylabel('Velocity (m/s)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_ylim([min(0, np.min(vel)*1.1), np.max(vel)*1.1])
        axes[1].ticklabel_format(style='plain', axis='y')

        #Acceleration vs time
        axes[2].plot(t, acc, 'r-', linewidth=2, label='Acceleration')
        axes[2].axhline(y=self.g0, color='k', linestyle='--', linewidth=1, label='1G (Earth gravity)')
        axes[2].set_ylabel('Acceleration (m/s^2)', fontsize=11)
        axes[2].set_xlabel('Time (seconds)', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_ylim([np.min(acc)*1.1, np.max(acc)*1.1])
        axes[2].ticklabel_format(style='plain', axis='y')
        #this helps judge if acceleration is reasonable

        plt.tight_layout()
        return fig

if __name__ == "__main__":
    """
    This runs when you execute the script directly: python rocket_simulator.py

    Execution Flow:
    ---------------
    1. Create a rocket simulator instance
    2. Run the simulation
    3. Print key statistics
    4. Generate and save visualizations
    5. Display plots

    This demonstrates the simulator's capabilities and validates results.
    """

    print("="*60)
    print("ROCKET TRAJECTORY SIMULATOR")
    print("="*60)

    #this create rocket simulator instance
    rocket = RocketSimulator(rocket_name="RFA Technologie Demonstrator")

    print(f"\nSimulating: {rocket.rocket_name}")
    print(f"Initial Mass: {rocket.m_empty + rocket.m_fuel + rocket.m_payload:0f}kg")
    print(f"Thrust: {rocket.thrust/1000:.1f}kN")
    print(f"Burn Time: {rocket.burn_time} seconds")
    print("\nRunning simulation...")

    #Simulates 600 seconds (10 minutes) of flight
    t, alt, vel, acc = rocket.simulate(t_max=600)

    print(f"\nDEBUG: velocity range: min={np.min(vel)}, max={np.max(vel)}")
    print(f"DEBUG: acceleration range: min={np.min(acc)}, max={np.max(acc)}")
    print(f"DEBUG: First 10 velocity values: {vel[:10]}")
    print(f"DEBUG: First 10 acceleration values: {acc[:10]}")

    max_alt = np.max(alt)
    max_vel = np.max(vel)
    max_acc = np.max(acc)

    burnout_idx = int(rocket.burn_time / 0.1)

    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Maximum Altitude:     {max_alt:.1f} km")
    print(f"Maximum Velocity:     {max_vel:.1f} m/s ({max_vel/1000:.2f} km/s)")
    print(f"Maximum Acceleration: {max_acc:.1f} m/s^2 ({max_acc/9.81:.1f} G's)")
    print(f"")
    print(f"At Burnout ({rocket.burn_time}s):")
    print(f" Altitude:     {alt[burnout_idx]:.1f} km")
    print(f" Velocity:     {vel[burnout_idx]:.1f} m/s")
    print(f" Acceleration: {acc[burnout_idx]:.1f} m/s^2")
    print("="*60)

    print("\nValidation:")
    if max_alt > 50 and max_alt < 200:
        print("Maximum altitude is realistic for small launcher")
    else:
        print("Maximum altitude seems unusual")

    if max_vel > 1000 and max_vel < 5000:
        print("Maximum Velocity is realistic")
    else:
        print("Maximum velocity seems unusual")

    if max_acc > 0 and max_acc < 100:
        print("Maximum acceleration is safe")
    else:
        print("Maximum acceleration may be too high")

    print("\nGenerating plots...")
    print(f"\nDEBUG INFO:")
    print(f"Length of t: {len(t)}")
    print(f"Length of alt: {len(alt)}")
    print(f"Length of vel: {len(vel)}")
    print(f"Length of acc: {len(acc)}")
    print(f"Velocity min: {np.min(vel)}, max: {np.max(vel)}")
    print(f"Acceleration min: {np.min(acc)}, max: {np.max(acc)}")
    print(f"First 5 acceleration values: {acc[:5]}")
    print(f"Last 5 acceleration values: {acc[-5:]}")
    
    fig = rocket.plot_trajectory(t, alt, vel, acc)

    output_filename = 'rocket_trajectory.png'
    plt.savefig(output_filename,dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")

    print("\nDisplaying plots...")
    plt.show()

    print("\nSimulation complete!")
    
