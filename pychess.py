"""
Usage:
    pychess.py process -i <in> -o <out>
    pychess.py analyze -i <in> -o <out> [--catmap <cat_file> --catthreshold <cat_thresh>]

Arguments:
    <in> (required) : path to input file or directory.  in can be one of:
        (a) directory of .csv or .xlsx files like a seva or eldertree export,
        (b) .yaml file pointing to urls where to load data (like for bundling), or
        (c) a .json or .pickle file containing the serialized study objects
            (e.g. for loading the output of process into analyze)
    <out> (required) : where to write the output json or pickle file
        Must have a `.json` or `.pickle` (or `.pkl`) extension. If `.json`, data will be serialized in a
        quasi-human-readable format that will be easier to load into other applications/languages/do whatever you want.
        If `.pickle`, data will be serialized using pickle, which will preserve all the data types and structures.
    <catmap> (optional) : a .yaml file that maps "action" categories to another category.
        For getting activities to correspond across studies (e.g. seva and bundling).
        See seva_to_bundling.yaml or reduce_bundling.yaml for examples.
        If no catmap supplied, categories are left as is.
    <cat_thresh> (optional) : a .yaml file that maps "action" categories to its "threshold".
        That is, how many times must this action be done in one period in order for that period to count as
        "active" for that category for that period.  If no catmap is supplied, thresholds will be automatically tuned
        to achieve a target inactivity rate. See Analysis options in this code for more details.

Options:
    -h, --help : print this menu

Example:
    > python pychess.py process -i seva_export/ -o seva_study.json
    > python pychess.py analyze -i seva_study.json -o seva_analysis.json

"""
try:
    import pandas as pd
    import requests
    from docopt import docopt
    import datetime as dt
    import json
    import warnings
    import os
    import yaml
    import numpy as np
    import pickle
    import itertools
    import sys
    from io import StringIO
except ImportError as e:
    print(e)
    print("The following packages are required:\n requests, docopt, pyyaml, numpy\n")
    sys.exit()
from config import *

# Debug mode (if True, doesn't load all the data, for speed)
DEBUG = False
DEBUG_N = 10000  # number of rows of each table to load, in debug mode

# Formatting options - probably don't need to change these
DATE_FORMAT = '%m/%d/%Y'
DATETIME_FORMAT = '%m/%d/%Y %H:%M:%S'


def main(args):
    if args['process']:
        process(args['<in>'], args['<out>'])
    if args['analyze']:
        analyze(args['<in>'], args['<out>'], args['<cat_file>'], args['<cat_thresh>'])


def process(file, out_file):

    study = Study()
    if os.path.isdir(file):
        study.load_directory(file)
    elif 'yaml' in file or 'yml' in file:
        study.load_url(file)
    else:
        print(__doc__)
        sys.exit()

    if '.json' in out_file:
        study.save_to_json(out_file)
    elif '.pickle' in out_file or '.pkl' in out_file:
        with open(out_file, 'wb') as f:
            pickle.dump(study, f)
    else:
        print("Choose an outfile with `.json` or `.pickle` extension.")

    return study


def analyze(study_file, out_file, cat_reduce_file, cat_threshold_file):

    if cat_reduce_file is None:
        cat_reduce_map = None
    else:
        with open(cat_reduce_file) as f:
            cat_reduce_map = yaml.load(f)
    if cat_threshold_file is None:
        cat_threshold_map = None
    else:
        with open(cat_threshold_file) as f:
            cat_threshold_map = yaml.load(f)

    if '.json' in study_file:
        study = Study()
        study.load_from_json(study_file)
    elif '.pickle' in study_file or '.pkl' in study_file:
        with open(study_file, 'rb') as f:
            study = pickle.load(f)
    else:
        print("Input file must have `.json` or `.pickle` extension.")

    analysis = Analysis(study, cat_reduce_map)

    if '.json' in out_file:
        analysis.save_to_json(out_file)
    elif '.pickle' in out_file or '.pkl' in out_file:
        with open(out_file, 'wb') as f:
            pickle.dump(analysis, f)
    else:
        print("Choose an outfile with `.json` or `.pickle` extension.")


class PychessObject:
    def __init__(self):
        self.dont_save = ['load', 'dont_save']
        self.load = []

    def to_json(self):
        return {str(k): make_json(v) for k, v in self.__dict__.items() if k not in self.dont_save}

    def from_json(self, d):
        for k in self.load:
            if k in d:
                self.__dict__[k] = unmake_json(d[k])
        return self


class Study(PychessObject):
    def __init__(self):
        super(Study, self).__init__()
        self.dont_save += ['user_by_id', 'user_by_username', 'cats']
        self.users = []
        self.user_by_id = dict()
        self.user_by_username = dict()
        self.cats = set()

    def add_user(self, user):
        self.users.append(user)
        self.user_by_id[user.id] = user
        self.user_by_username[user.username] = user

    def get_cats(self):
        """
        Returns a list of the categories. Sorted alphabetically for consistency.
        """
        return sorted(list(self.cats))

    def get_cat_id(self, cat):
        cats = self.get_cats()
        if cat in cats:
            return cats.index(cat)
        else:
            return None

    def load_url(self, url_file):

        with open(url_file) as f:
            urls = yaml.load(f)

        user_data = requests.get(urls['user_data']).text
        user_data = pd.read_csv(StringIO(user_data)).fillna('')

        if 'use_data' in urls:
            use_data = requests.get(urls['use_data']).text
            use_data = use_data.splitlines()
            # Some lines have too many columns - filter out/ignore them
            n_columns = len(use_data[0].split(','))
            use_data = '\n'.join([u for u in use_data if len(u.split(',')) == n_columns])
            use_data = pd.read_csv(StringIO(use_data)).fillna('')
        else:
            use_data = None

        self.load_from_dataframe(user_data, use_data)

    def load_directory(self, directory):

        all_files = next(os.walk(directory))[2]
        user_files = [os.path.join(directory, f) for f in all_files if 'users' in f]
        use_files = [os.path.join(directory, f) for f in all_files if 'use-data' in f]
        discussion_files = [os.path.join(directory, f) for f in all_files if 'discussion-messages' in f]
        private_files = [os.path.join(directory, f) for f in all_files if 'private-messages' in f]

        user_data = pd.DataFrame()
        use_data = pd.DataFrame()
        discussion_data = pd.DataFrame()
        private_data = pd.DataFrame()

        for user_file in user_files:
            if user_file[-4:] == 'xlsx':
                user_data = user_data.append(pd.read_excel(user_file), ignore_index=True)
            else:
                user_data = user_data.append(pd.read_csv(user_file, encoding='latin1'), ignore_index=True)
        for use_file in use_files:
            if use_file[-4:] == 'xlsx':
                use_data = use_data.append(pd.read_excel(use_file), ignore_index=True)
            else:
                use_data = use_data.append(pd.read_csv(use_file, encoding='latin1'), ignore_index=True)
        for discussion_file in discussion_files:
            if discussion_file[-4:] == 'xlsx':
                discussion_data = discussion_data.append(pd.read_excel(discussion_file), ignore_index=True)
            else:
                discussion_data = discussion_data.append(pd.read_csv(discussion_file, encoding='latin1'),
                                                         ignore_index=True)
        for private_file in private_files:
            if private_file[-4:] == 'xlsx':
                private_data = private_data.append(pd.read_excel(private_file), ignore_index=True)
            else:
                private_data = private_data.append(pd.read_csv(private_file, encoding='latin1'), ignore_index=True)

        self.load_from_dataframe(user_data, use_data, discussion_data, private_data)

    def load_from_dataframe(self, user_data, use_data=None, discussion_data=None, private_data=None):

        user_data = user_data.fillna('')
        i = 0
        for row in user_data.iterrows():
            user_info = dict(row[1])
            user_info = {k.lower().strip(): v for k, v in user_info.items()}
            user = User(**user_info)
            self.add_user(user)

            i += 1
            if DEBUG and i > DEBUG_N:
                break

        if use_data is not None:
            use_data = use_data.fillna('')
            i = 0
            for row in use_data.iterrows():
                action_info = dict(row[1])
                action_info = {k.lower().strip(): v for k, v in action_info.items()}
                if 'actiontag' in action_info:
                    cat = '{}/{}'.format(action_info['servicetag'], action_info['actiontag'])
                    cat = cat.lower()
                elif 'page' in action_info:
                    cat = action_info['page']
                else:
                    cat = action_info['url']

                cat = clean_cat(cat)

                self.cats.add(cat)

                action_info['cat'] = cat
                action = Action(**action_info)
                if action.username in self.user_by_username:
                    self.user_by_username[action.username].add_action(action)

                i += 1
                if DEBUG and i > DEBUG_N:
                    break

        if discussion_data is not None:
            discussion_data = discussion_data.fillna('')
            i = 0
            for row in discussion_data.iterrows():
                msg_info = dict(row[1])
                msg_info = {k.lower().strip(): v for k, v in msg_info.items()}
                msg = ForumMsg(**msg_info)
                if msg.username in self.user_by_username:
                    self.user_by_username[msg.username].forum_msgs.append(msg)

                i += 1
                if DEBUG and i > DEBUG_N:
                    break

        if private_data is not None:
            i = 0
            private_data = private_data.fillna('')
            for row in private_data.iterrows():
                msg_info = dict(row[1])
                msg_info = {k.lower().strip(): v for k, v in msg_info.items()}
                msg = PrivateMsg(**msg_info)
                if msg.author in self.user_by_username:
                    self.user_by_username[msg.author].private_msgs.append(msg)

                i += 1
                if DEBUG and i > DEBUG_N:
                    break

    def from_json(self, d):
        for u in d['users']:
            self.add_user(User().from_json(u))
        for u in self.users:
            for a in u.actions:
                self.cats.add(a.cat)
        return self

    def save_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_json(), f)

    def load_from_json(self, filename):
        with open(filename) as f:
            self.from_json(json.load(f))


class User(PychessObject):
    def __init__(self, **kwargs):
        super(User, self).__init__()
        self.load = ['id', 'username', 'studyid', 'gender', 'role', 'agency', 'status', 'joindate',
                     'first_action_timestamp', 'demo_ended', 'active']

        if 'userid' in kwargs:
            self.id = kwargs['userid']
        if 'username' in kwargs:
            self.username = kwargs['username']
        if 'studyid' in kwargs:
            self.studyid = kwargs['studyid']
        if 'gender' in kwargs:
            self.gender = kwargs['gender']
        if 'role' in kwargs:
            self.role = kwargs['role']
        if 'agency' in kwargs:
            self.agency = kwargs['agency']
        if 'status' in kwargs:
            self.status = kwargs['status']
        if 'joindate' in kwargs:
            self.joindate = parse_date(kwargs['joindate'])
        elif 'enrollmentdate' in kwargs:
            self.joindate = parse_date(kwargs['enrollmentdate'])

        self.actions = []
        self.first_action_timestamp = None
        self.demo_ended = None
        self.forum_msgs = []
        self.private_msgs = []
        self.active = None

    def add_action(self, action):
        self.actions.append(action)
        if action.date is not None:
            if self.first_action_timestamp is None:
                self.first_action_timestamp = action.date.timestamp()
            elif action.date.timestamp() < self.first_action_timestamp:
                self.first_action_timestamp = action.date.timestamp()

    def from_json(self, d):
        self.actions = [Action().from_json(a) for a in d['actions']]
        self.forum_msgs = [ForumMsg().from_json(fm) for fm in d['forum_msgs']]
        self.private_msgs = [PrivateMsg().from_json(pm) for pm in d['private_msgs']]

        return super(User, self).from_json(d)


class Action(PychessObject):
    def __init__(self, **kwargs):
        super(Action, self).__init__()
        self.load = ['user_id', 'username', 'page', 'seconds_spent', 'date', 'cat']

        if 'userid' in kwargs:
            self.user_id = kwargs['userid']
        if 'username' in kwargs:
            self.username = kwargs['username']
        elif 'visitor' in kwargs:
            self.username = kwargs['visitor']
        if 'visitdate' in kwargs:
            self.date = parse_date(kwargs['visitdate'])
        if 'date' in kwargs:
            self.date = parse_date(kwargs['date'])
        if 'page' in kwargs:
            self.page = kwargs['page']
        if 'cat' in kwargs:
            self.cat = kwargs['cat']
        if 'secondspentonpage' in kwargs:
            self.seconds_spent = kwargs['secondsspentonpage']

    def from_json(self, d):
        return super(Action, self).from_json(d)


class PrivateMsg(PychessObject):
    def __init__(self, **kwargs):
        super(PrivateMsg, self).__init__()
        self.load = ['message_id', 'user_id', 'author', 'date', 'receiver', 'seen']

        if 'msgid' in kwargs:
            self.message_id = kwargs['msgid']
        if 'userid' in kwargs:
            self.user_id = kwargs['userid']
        if 'author' in kwargs:
            self.author = kwargs['author']
        if 'datewritten' in kwargs:
            self.date = parse_date(kwargs['datewritten'])
        if 'threadparticipants' in kwargs:
            participants = kwargs['threadparticipants'].split(', ')
            receiver = [u for u in participants if u != self.author]
            if len(receiver) > 0:
                self.receiver = receiver[0]
            else:
                self.receiver = self.author
        if 'visitedbyboththreadparticipants' in kwargs:
            self.seen = bool(kwargs['visitedbyboththreadparticipants'])


class ForumMsg(PychessObject):
    def __init__(self, **kwargs):
        super(ForumMsg, self).__init__()
        self.load = ['topic_name', 'parent_id', 'comment_id', 'user_id', 'username', 'type', 'title', 'message',
                     'date', 'isanonymous']

        if 'topicname' in kwargs:
            self.topic_name = kwargs['topicname']
        if 'parentmessageguid' in kwargs:
            self.parent_id = kwargs['parentmessageguid']
        if 'commentmessageguid' in kwargs:
            self.comment_id = kwargs['commentmessageguid']
        if 'userid' in kwargs:
            self.user_id = kwargs['userid']
        if 'author' in kwargs:
            self.username = kwargs['author']
        if 'message' in kwargs:
            self.type = kwargs['message']
        if 'messagetitle'  in kwargs:
            self.title = kwargs['messagetitle']
        self.message = ''
        if 'message' in kwargs:
            self.message = kwargs['message']
        if 'comment' in kwargs and self.message == '':
            self.message = kwargs['comment']
        if 'datewritten' in kwargs:
            self.date = parse_date(kwargs['datewritten'])
        if 'isanonymous' in kwargs:
            self.anonymous = bool(kwargs['isanonymous'])


class Analysis(Study):
    def __init__(self, study, cat_reduce_map=None, cat_threshold_map=None):
        super(Analysis, self).__init__()

        if DEBUG:
            self.users = study.users
        else:
            for u in study.users:
                if not is_admin(u):
                    self.users.append(u)

        if cat_reduce_map is not None:
            old_cats = study.cats
            for cat in old_cats:
                if cat in cat_reduce_map and cat_reduce_map[cat] != 'null':
                    self.cats.add(cat_reduce_map[cat])
                else:
                    print("No map for category: {}... not including in analysis".format(cat))
            for u in self.users:
                new_actions = []
                for action in u.actions:
                    if action.cat in cat_reduce_map:
                        action.cat = cat_reduce_map[action.cat]
                        new_actions.append(action)
                        self.cats.add(action.cat)
                    elif action.cat in self.cats:
                        new_actions.append(action)
                u.actions = new_actions
        else:
            self.cats = study.cats

        cats = self.get_cats()
        cats_set = set(cats)

        num_cats = len(cats)
        num_periods = MAX_LENGTH // PERIOD_LENGTH + 1

        for u in self.users:
            if u.demo_ended is None and u.first_action_timestamp is not None:
                u.demo_ended = u.first_action_timestamp + DEMO_LENGTH
            else:
                u.demo_ended = 0
            u.history = np.zeros((num_cats, num_periods))
            u.activity = np.zeros((num_cats, num_periods))
            for a in u.actions:
                if a.cat in cats_set and a.date is not None:
                    relative_time = a.date.timestamp() - u.first_action_timestamp
                    if relative_time < MAX_LENGTH and a.date.timestamp() > u.demo_ended:
                        period = int(relative_time // PERIOD_LENGTH)
                        u.history[cats.index(a.cat), period] += 1

            if USE_SMOOTH_HISTORY:
                u.history = moving_average(u.history, SMOOTHING_WINDOW)

            u.train_history = u.history[:, TRAINING_PERIODS]
            u.predict_history = u.history[:, PREDICT_PERIODS]

        if cat_threshold_map is None:
            cat_threshold_map = dict()
            cat_period_threshold_map = dict()

            threshold_search = np.logspace(-2, 0, 50)
            num_period_search = range(1, len(PREDICT_PERIODS) + 1)
            grid_search = list(itertools.product(threshold_search, num_period_search))

            for i, cat in enumerate(self.get_cats()):

                eval_map = dict()
                for cat_t, num_p in grid_search:
                    user_inactivity_map = {u.id: 0 for u in self.users}
                    for u in self.users:
                        periods_this_cat = u.predict_history[i, :]
                        for p in periods_this_cat:
                            if u.id in user_inactivity_map:
                                if p < cat_t:
                                    user_inactivity_map[u.id] += 1
                                    if user_inactivity_map[u.id] >= num_p:
                                        del user_inactivity_map[u.id]
                            else:
                                break

                    percent_active = len(user_inactivity_map) / len(self.users)
                    eval_map[(cat_t, num_p)] = 1 - percent_active

                best_t = sorted(eval_map.keys(), key=lambda x: abs(eval_map[x] - TARGET_INACTIVITY_RATE))[0]

                cat_threshold_map[cat] = best_t[0]
                cat_period_threshold_map[cat] = best_t[1]
        else:
            cat_period_threshold_map = {c: DEFAULT_INACTIVITY_PERIOD_THRESHOLD for c in self.get_cats()}

        cat_period_threshold_vector = np.array([cat_period_threshold_map[c] for c in self.get_cats()])

        # Classify active and inactive users
        for u in self.users:
            for j in range(len(PREDICT_PERIODS)):
                for i, c in enumerate(self.get_cats()):
                    if cat_threshold_map[c] > 0:
                        u.activity[i, j] = int(u.predict_history[i, j] > cat_threshold_map[c])

            activity_thresholded = (u.activity.sum(axis=1)) > cat_period_threshold_vector
            u.num_active_cats = activity_thresholded.sum()

        if NUM_CATS_THRESHOLD is None:
            candidate_num_cats = range(1, len(self.get_cats()) + 1)
            eval_map = {}
            for c in candidate_num_cats:
                inactive_users = [u for u in self.users if u.num_active_cats < c]
                eval_map[c] = len(inactive_users) / len(self.users)

            num_cats_threshold = sorted(eval_map.keys(), key=lambda x: abs(eval_map[x] - TARGET_INACTIVITY_RATE))[0]
        else:
            num_cats_threshold = NUM_CATS_THRESHOLD

        for u in self.users:
            if u.num_active_cats < num_cats_threshold:
                u.active = 0
            else:
                u.active = 1


def is_admin(user):
    """
    Function designed to filter out admins
    """
    if user.joindate is None:
        return True
    if (dt.datetime.now() - user.joindate).days > 999:
        return True
    if any([f in user.username.lower() for f in ADMIN_FLAGS]):
        return True
    if len(user.actions) == 0:
        return True
    return False


def moving_average(x, window, pad=False):
    '''
    Returns moving average of 1d vector x with window size N.
    If x is a matrix, returns a matrix of moving average of each row.
    If pad is true, will fill with 0s to make the output vector the same length as input.
    '''
    if pad:
        pad_left = window // 2
        pad_right = window // 2
        if window % 2 == 0 and pad_left > 0:
            pad_left -= 1
    else:
        pad_left = 0
        pad_right = 0

    for i in range(x.shape[0]):
        cumsum = np.cumsum(np.insert(x[i, :], 0, 0))
        if i == 0:
            output = np.zeros((x.shape[0], (np.hstack((np.zeros(pad_left), (cumsum[window:] - cumsum[:-window]) / window, np.zeros(pad_right)))).shape[0]))
        output[i, :] = np.hstack((np.zeros(pad_left), (cumsum[window:] - cumsum[:-window]) / window, np.zeros(pad_right)))

    return output


def clean_cat(cat):
    for ch in COMBINE_CATS:
        if ch in cat:
            return ch

    return cat


def parse_date(date):
    if date is None or date == '':
        return None
    elif isinstance(date, dt.datetime):
        return date
    elif isinstance(date, pd.Timestamp):
        return dt.datetime.combine(date.date(), date.time())
    elif isinstance(date, dt.date):
        return dt.datetime.combine(date, dt.datetime.min.time())
    elif isinstance(date, str):
        try:
            return dt.datetime.strptime(date, DATETIME_FORMAT)
        except ValueError:
            try:
                return dt.datetime.strptime(date, DATE_FORMAT)
            except ValueError:
                warnings.warn('Date: ' + str(date) + ' could not be parsed')
                return None


def make_json(obj):
    if obj is None:
        return None
    if hasattr(obj, 'to_json'):
        return obj.to_json()
    if isinstance(obj, (str, int, float)):
        return obj
    if isinstance(obj, dt.datetime):
        return dt.datetime.strftime(obj, DATETIME_FORMAT)
    if isinstance(obj, list):
        return [make_json(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): make_json(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def unmake_json(obj):
    if obj is None:
        return obj

    try:
        return dt.datetime.strptime(obj, DATETIME_FORMAT)
    except ValueError:
        pass
    except TypeError:
        pass

    '''
    try:
        return float(obj)
    except ValueError:
        pass
    except TypeError:
        pass
    '''

    return obj


if __name__ == '__main__':
    main(docopt(__doc__))