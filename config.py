###################
# Analysis options
###################

# Cats that start with these will be combined to the same category
#  e.g.
# "Q & A: 5" -> "Q & A"
# "Q & A: 7" -> "Q & A"
# "Article - 10" -> "Article"
# etc.
COMBINE_CATS = ['Q & A', 'Recovery Motivation', 'Article', 'Discussion Group', 'Quick Tip', 'Admin']

# usernames (after converted to lowercase) containing these strings will be filtered out as admins:
ADMIN_FLAGS = ['uw', 'tech']

# Lengths are in seconds
# Length of periods to bin activities into
PERIOD_LENGTH = 60 * 60 * 24 * 7  # 1 week
# Seconds to ignore due to "training time"
# TODO apparently Fiona has the exact training times somewhere...
# TODO You can set this (user.demo_ended = <datetime object>.timestamp) and it is supported (and has precedence)
DEMO_LENGTH = 60 * 40  # 40 minutes
# Time after which events are ignored
MAX_LENGTH = 60 * 60 * 24 * 7 * 60  # 60 weeks

# Use a moving average for the history matrix?
USE_SMOOTH_HISTORY = False
# If so, how many periods of a window?
SMOOTHING_WINDOW = 5

# These are the periods used for "training"
TRAINING_PERIODS = [0, 1, 2]
# This is the window over which you are trying to predict activity
PREDICT_PERIODS = [3]
# Thresholds for categories will be tuned to hit this inactivity rate
# Has no effect if you supply your own cat_threshold_map
TARGET_INACTIVITY_RATE = .2
# How many periods in "predict_periods" someone has to be inactive to be classified "inactive"
# Only matters if you supply your own cat_threshold_map...
# Otherwise, this parameter will get tuned to achieve TARGET_INACTIVITY_RATE
DEFAULT_INACTIVITY_PERIOD_THRESHOLD = 1
# How many categories must a user be active in to be considered active?
# Set to None to tune this to achieve TARGET_INACTIVITY_RATE
NUM_CATS_THRESHOLD = None

#############