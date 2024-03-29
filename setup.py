from setuptools import setup, find_packages

# Install with 'pip install -e .'

setup(
    name="gym_gfetch",
    version="0.1.0",
    author="Alexandre Chenu, Nicolas Perrin-Gilbert",
    description="Robot manipulation environment for goal-conditioned RL (with OpenAI Gym interface)",
    url="https://github.com/AlexandreChenu/gfetch_dcil",
    packages=find_packages(),
    install_requires=[
        "gym>=0.22.0",
        "torch>=1.10.0",
        "matplotlib>=3.1.3",
    ],
    license="LICENSE",
)
