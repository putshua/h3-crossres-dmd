"""List registered experiments and default groups without loading models."""

from h3dmd.configs import experiment_names
from h3dmd.registry import CONFIG_GROUPS


def main():
    print("Experiments:")
    for name in experiment_names():
        print(f"  {name}")
    print("Config groups:")
    for name in CONFIG_GROUPS.names():
        print(f"  {name}")


if __name__ == "__main__":
    main()
