# CEE6501-Project
This is the final project for CEE 6501 Matrix Structural Analysis which will implement the Direct Stiffness Method (DSM) into Python.

## Features
- 3D Truss Elements
- 3D Frame elements
- Mixed element types
- JSON input file
- Ouputs containing displacements, reactions, and forces as well as figures

## How to run
1. Open 'main.ipynb'
2. Modify input name in quotes "" of either "example1", "example2", "example2_supports', "example2_temperature", "example2_fab_err", "structure", "case_1_study", "case_2_study", "user_added"
3. Run cell by clicking shift+enter

## Structure
- The helpers folder will contain the main function where all the calculations happens
- The inputs folder will contain the models I will be using or any that the user wants to add
- The outputs folder will store the results in a .json file along with a .png of the deformed results
- The report folder will store the final report
