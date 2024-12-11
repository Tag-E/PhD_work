from Kfact_calculator import K_calc

Ks = K_calc((4,1),(4,1))
Ks.latex_print()

Ks_A = K_calc((4,1),(4,4))
Ks_A.latex_print()

Ks_2 = K_calc((4,1),(4,1),(4,1))
Ks_2.latex_print()