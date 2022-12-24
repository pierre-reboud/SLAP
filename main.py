#!/home/pierre/envs/general/bin/python
from slap.utils.utils import Configs
from slap.slam import Slam

if __name__ == "__main__":
    configs = Configs()
    slam = Slam(configs)
    slam._test_run()
    #slam.run()
