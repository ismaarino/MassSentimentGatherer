import json
import globals as g
import gather


def init():
	config = json.load(open(g.CONFIG_FILE))
	gather.run(config)
	

init()
