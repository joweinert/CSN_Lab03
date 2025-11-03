# CSN_Lab03

To run the project follow these steps:

1. create a new virtual environment in the project root: "python -m venv venv"
2. activate it: ".\venv\Scripts\activate" (Windows version)
3. install the requirements: "pip install -r requirements.txt"
4. run "python -m src.monte_carlo --T {int} --Q {int}" (If needed configure --closeness_fn and --nullmodel) 

monte_carlo.py is the maion script. It holds the closness calculation candidates, the monte carlo logic, ...
closeness_benchmark.py is interesting to understand the benchmark procedure.
null_model.py is for nullmodel instantiatiuon

Notebook lab03 is home to some of the described experiments (forn instance the different Qs for T=20 experiment)
Notebook analysis is a rather short collection of quick answers to immediate questions