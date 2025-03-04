from cgh4_calculator import cg_calc
cg = cg_calc((4,1),(4,1), prescription_changed=True)
cg.latex_print()

cg_a = cg_calc((4,4),(4,1), prescription_changed=True)
cg_a.latex_print()

cg2 = cg_calc((4,1),(4,1),(4,1), prescription_changed=True)
cg2.latex_print()

cg2_a = cg_calc((4,4),(4,1),(4,1), prescription_changed=True)
cg2_a.latex_print()

#cg3 = cg_calc((4,1),(4,1),(4,1),(4,1), prescription_changed=True)
#cg3.latex_print()