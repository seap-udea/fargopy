import fargopy as fp
import os


# Mock 2D simulation
def mock_2d():
    print("Loading 2D sim...")
    # This should modify the default borders if the bug exists
    # We don't need real files if we can trigger the function logic
    # But _load_domains needs file access.
    # Typically we can rely on the fact that the user already ran it.
    pass


# We check the default value of _load_domains
print(f"Default borders before: {fp.Simulation._load_domains.__defaults__[0]}")

# Since we cannot easily modify the live environment, let's verify the source logic visually first.
# The previous view_file confirmed:
# def _load_domains(self, vars, domain_prefix, borders=[[], [3, -3], [3, -3]], middle=True):
#    if vars.DIM == 2:
#       borders[-1] = []

# This confirms the bug.
# I will proceed to fix it directly.
