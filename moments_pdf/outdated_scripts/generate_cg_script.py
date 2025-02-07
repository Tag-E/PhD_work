import sys #to fetch command line argument
from cgh4_calculator import cg_calc #to test the cg generator


#number of indices reading from command line
n=2 #std value
if len(sys.argv) > 1:
    try:
        n = int(sys.argv[1])
    except ValueError:
        print(f"\nSpecified number of indices n was {sys.argv[1]}, as it cannot be casted to int we proceed with n={n}\n")
#reading which kind of computation to do
which = "both"
if len(sys.argv) > 2:
    if str(sys.argv[2]) in ["both","vector","axial"]:
        which = str(sys.argv[2])
    else:
        print(f"\nSpecified type of computation was {sys.argv[2]} but a choice must be made between 'both','vector', 'axial'. Computation will be done with 'both'\n")


#consequent selection of irreps to use
chosen_irreps = [(4,1)]
chosen_irreps_axial = [(4,4)]
while len(chosen_irreps) < n:
    chosen_irreps.append((4,1))
    chosen_irreps_axial.append((4,1))


#cg generation
if which=="both" or which=="vector":
    cg = cg_calc(*chosen_irreps, force_computation=True)
    cg.latex_print()
if which=="both" or which=="axial":
    cg_a = cg_calc(*chosen_irreps_axial, force_computation=True)
    cg_a.latex_print()