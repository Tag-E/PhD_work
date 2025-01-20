import os #to fetch env variables
import sys #to fetch command line argument
from moments_toolkit import moments_toolkit #to test the new library


#nconf read from command line
nconf=10 #std value
if len(sys.argv) > 1:
    try:
        nconf = int(sys.argv[1])
    except ValueError:
        print(f"\nSpecified nconf was {sys.argv[1]}, as it cannot be casted to int we proceed with nconf={nconf}\n")


p3fold = os.environ['mount_point_path'] + "48c48/binned_1012_hmz370_BMW/3PointCorrelation/"
p2fold = os.environ['mount_point_path'] + "48c48/binned_1012_hmz370_BMW/2PointCorrelation/"


momAn = moments_toolkit(p3fold,p2fold,maxConf=nconf,verbose=True)



momAn.operator_show(show=False, verbose=True)

momAn.select_operator(28,32)


momAn.plot_R(save=True,show=False)

print("\nS mean and std: ")
_, S1mean, S1std = momAn.get_S(1)
print(S1mean, S1std)
_, S2mean, S2std = momAn.get_S(2)
print(S2mean, S2std)
_, S3mean, S3std = momAn.get_S(3)
print(S3mean, S3std)