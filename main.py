#!/home/pierre/envs/general/bin/python
from slap.utils.utils import Configs
from slap.slam import Slam
from slap.utils.utils import signal_handler 
import logging
from logging import info
import signal

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    signal.signal(signal.SIGINT, signal_handler)
    configs = Configs()
    slam = Slam(configs)
    slam.run()
    #slam.run()
