#!/home/pierre/envs/general/bin/python
from slap.utils.utils import Configs
from slap.slam import Slam
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    configs = Configs()
    slam = Slam(configs)
    slam._test_run()
    #slam.run()
