from cgh4_calculator import cg_calc
cg = cg_calc((4,1),(4,1), force_computation=True)
cg.latex_print()

cg_a = cg_calc((4,1),(4,4), force_computation=True)
cg_a.latex_print()

cg2 = cg_calc((4,1),(4,1),(4,1), force_computation=True)
cg2.latex_print()