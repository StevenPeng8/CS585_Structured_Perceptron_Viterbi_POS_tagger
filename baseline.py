from __future__ import division

import structperc

from collections import Counter

data = structperc.read_tagging_file('oct27.dev')

tags = reduce(lambda x,y: x + y, map(lambda x: x[1], data))

most_common = Counter(tags).most_common(1)[0]

print "Most common tag: " + most_common[0]

print "Accuracy Assuming all tags are " + most_common[0] + ": " + str(most_common[1] / len(tags))
