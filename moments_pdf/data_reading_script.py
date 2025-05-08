#simple script to load the relevant datasets

#library imports
import os
from moments_toolkit import moments_toolkit

#paths for the coarse lattice
p3fold = os.environ['mount_point_path'] + "48c48/binned_1012_hmz370_BMW_extended/3PointCorrelation/"
p2fold = os.environ['mount_point_path'] + "48c48/binned_1012_hmz370_BMW_extended/2PointCorrelation/"

# coarse lattice - P = 0
opAnalyzer_coarse_P0 = moments_toolkit(p3fold, p2fold,
                                        skip3p=False, skipop=False,
                                        verbose=True,
                                        fast_data_folder = "fast_data_extended_p0_q0",
                                        operator_folder= "operator_database",
                                        momentum='PX0_PY0_PZ0',
                                        insertion_momentum = 'qx0_qy0_qz0',
                                        smearing_index=0,
                                        tag_2p='hspectrum',
                                        max_n=2 #max_n=3
                                        )

# coarse lattice - Px = -2
opAnalyzer_coarse_Px = moments_toolkit(p3fold, p2fold,
                                        skip3p=False, skipop=False,
                                        verbose=True,
                                        fast_data_folder = "fast_data_extended_px-2_q0",
                                        operator_folder= "operator_database",
                                        momentum='PX-2_PY0_PZ0',
                                        insertion_momentum = 'qx0_qy0_qz0',
                                        smearing_index=0,
                                        tag_2p='hspectrum',
                                        max_n=2 #max_n=3
                                        )


#paths for the fine lattice
p3fold = os.environ['mount_point_path'] + "64c64/3pt_binned_h5/"
p2fold = os.environ['mount_point_path'] + "64c64/2pt_binned_h5/"

# fine lattice - P = 0
opAnalyzer_fine_P0 = moments_toolkit(p3fold, p2fold,
                                        skip3p=False, skipop=False,
                                        verbose=True,
                                        fast_data_folder = "fast_data_extended_fine_p0_q0",
                                        operator_folder= "operator_database",
                                        momentum='PX0_PY0_PZ0',
                                        insertion_momentum = 'qx0_qy0_qz0',
                                        smearing_index=0,
                                        tag_2p='hspectrum',
                                        max_n=2 #max_n=3
                                        )

# fine lattice - Px = -1
opAnalyzer_fine_Px = moments_toolkit(p3fold, p2fold,
                                        skip3p=False, skipop=False,
                                        verbose=True,
                                        fast_data_folder = "fast_data_extended_fine_px-1_q0",
                                        operator_folder= "operator_database",
                                        momentum='PX-1_PY0_PZ0',
                                        insertion_momentum = 'qx0_qy0_qz0',
                                        smearing_index=0,
                                        tag_2p='hspectrum',
                                        max_n=2 #max_n=3
                                        )

#paths for the coarse lattice needed for 2der operators
p3fold = os.environ['mount_point_path_newdataset'] + "48c48/binned20250430_hmz370_BMW_3.31_48c48_ml-0.09933_mh-0.04_connected_himom//3PointCorrelation/"
p2fold = os.environ['mount_point_path_newdataset'] + "48c48/binned20250430_hmz370_BMW_3.31_48c48_ml-0.09933_mh-0.04_connected_himom/2PointCorrelation/"

# coarse lattice - P = -2
opAnalyzer_coarse_2DER = moments_toolkit(p3fold, p2fold,
                                            skip3p=False, skipop=False,
                                            verbose=True,
                                            fast_data_folder = "fast_data_2DERdataset_p2",
                                            operator_folder= "operator_database",
                                            momentum='PX2_PY2_PZ2',
                                            insertion_momentum = 'qx0_qy0_qz0',
                                            smearing_index=1,
                                            tag_2p='hspectrum',
                                            max_n=3
                                            )


#paths for the coarse lattice needed for 2der operators
p3fold = os.environ['mount_point_path_newdataset'] + "64c64/binned20250502_hmz370_BMW_3.5_64c64_ml-0.05294_mh-0.006_connected_himom/3PointCorrelation/"
p2fold = os.environ['mount_point_path_newdataset'] + "64c64/binned20250502_hmz370_BMW_3.5_64c64_ml-0.05294_mh-0.006_connected_himom/2PointCorrelation/"

# coarse lattice - P = -2
opAnalyzer_fine_2DER = moments_toolkit(p3fold, p2fold,
                                            skip3p=False, skipop=False,
                                            verbose=True,
                                            fast_data_folder = "fast_data_2DERdataset_fine_p2",
                                            operator_folder= "operator_database",
                                            momentum='PX2_PY2_PZ2',
                                            insertion_momentum = 'qx0_qy0_qz0',
                                            smearing_index=1,
                                            tag_2p='hspectrum',
                                            max_n=3
                                            )

#we put all the class instances in one dictionary such that they can be accesed later
dataset_analyzer_dict = {
    "coarse_P0" : opAnalyzer_coarse_P0,
    "coarse_Px" : opAnalyzer_coarse_Px,
    "fine_P0" : opAnalyzer_fine_P0,
    "fine_Px" : opAnalyzer_fine_Px,
    "coarse_2der" : opAnalyzer_coarse_2DER,
    "fine_2der" : opAnalyzer_fine_2DER,
}