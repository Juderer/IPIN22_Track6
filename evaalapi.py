################################################################
## evaalapi.py
## a web API interface for running the IPIN competitions
## Copyright 2021-2022 Francesco Potortì
## This program can be freely run, copied, modified and distributed
## under a GNU Affero General Public License v3.0 or later
## SPDX-License-Identifier: AGPL-3.0-or-later
################################################################

import os
import lzma
import time  # sleep
from datetime import datetime, timezone  # strftime with microseconds
import yaml
from parse import parse

from markdown import markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

from flask import Flask, abort, request, Request, Response, send_file
from markupsafe import escape

revision = "$Revision: 3.7 $"[11:-2]
source = "evaalapi.py"
sourcedir = "source/"
trialsdir = "trials/"
globinifn = "evaalapi.yaml"
privatefn = "private.yaml"
starttime = datetime.now(timezone.utc).strftime("%FT%H:%M:%SZ")
guisource = "evaalapigui.py"

## The Flask app
app = Flask(__name__)

## This is crude, probably something should be done with the app.logger
debug = app.debug

## Useful for logging to get the originating IP when behind proxies
## In the Request object request, use X-Forwarded-For to set remote_addr
proxies = 1  # number of proxies
if proxies > 0:
    try:  # get remote ip when behind a proxy
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, proxies)
    except ImportError:
        pass

states = ('nonstarted', 'running', 'finished', 'timeout')
esthdr = "pts,c,h,s,pos"
estfmt = "{pts:.3f},{c:.3f},{h:.3f},{s:.3f},{pos}"
statefmt = "{trialts:.3f},{rem:.3f},{V:.3f},{S:.3f},{p:.3f},{h:.3f},{pts:.3f},{pos}"

trials = {}  # the known trials
test_trial = {"test":  # a trial used by the test suite
                  {'name': "test",
                   'datafile': "test.csv",  # text data file
                   'commsep': '%',  # comment-starting string in data file
                   'sepch': ',',  # field separator in data file
                   'V': 1,  # normally 3, but testsuite would be slower
                   'S': 2,  # normally 15, but testsuite would be slower
                   'inipos': "10.422057,43.718278,1",  # my office location in WGS84
                   'reloadable': True}}

## OpenAPI description initialisation using the swagger library
##
## A bug in flasgger 0.9.51 keeps newlines in description even with MK_SANITIZER
## so we should remove the newlines from the string.  We do it by using yaml
## block scalars (see https://yaml-multiline.info/).  Note that the blank lines
## between paragraphs are indented: the number of space characters matters!
apidescription = """
    ## IPIN competition interface

    For official use at the [IPIN competition](http://evaal.aaloa.org) you need a trial name.

    For unofficial testing you can freely use the `demo` trial, either by
    writing your own tests or by running the [demo](/evaalapi/demo)
    program at your premises.  Calling `demo auto` produces [this output](/evaalapi/demo-auto.out)
    on your terminal.  Calling `demo interactive` allows one to choose the timing by
    pressing Return at the terminal.

    If you want to run `demo` with an API server at your premises, you need to download
    the API server source code and the [demo configuration](/evaalapi/evaalapi.yaml) in the
    same directory, plus the `Logfiles/01-Training/01a-Regular/T03_02.txt` file taken from
    [Indoorloc](http://indoorloc.uji.es/ipin2020track3/files/logfiles2020.zip), to be put
    under a `trials/` subdirectory.

    ### Documentation

    You can find a browsable, interactive [OpenAPI description](/evaalapi/apidocs/)
    and the API [complete documentation](/evaalapi/evaalapi.html) (Markdown [source](/evaalapi/evaalapi.md)).

    ### Source code

    Copyright 2021-2022 Francesco Potortì

    The Python [""" + source + """](/evaalapi/""" + source + """) source code is released under the
    [GNU Affero General Public License v3.0 or later](https://www.gnu.org/licenses/agpl-3.0.html).
"""
swagger_config = {
    ## See examples/changelog_090.py for additional customisations
    'title': "EvaalAPI",
    'uiversion': '3',  # 3 is the default
    'swagger_ui': True,
    'hide_top_bar': True,  # hide the green top bar
    ## 'top_text': "<h1>A web API for the IPIN competition</h1><br><hr>",
    'footer_text': "<b>Copyright 2021 Francesco Potortì - GNU AGPL 3.0 or later",
    'headers': [],
    'specs': [{'endpoint': "openapispec",
               'route': "/evaalapi/openapispec.json",
               'rule_filter': lambda rule: True,  # all in
               'model_filter': lambda tag: True,  # all in
               }],
    'static_url_path': "/evaalapi/flasgger_static",
    'specs_route': "/evaalapi/apidocs/"
}
swagger_template = yaml.safe_load("""
info:
  title: EvaalAPI
  schemes:
   - https
  description: >
""" + apidescription + """
  contact:
   name: Francesco Potortì
   email: Potorti@isti.cnr.it
   url: http://fly.isti.cnr.it
  version: "1.0"
    ## The definition of Trialname is not conformant to OpenAPI 2.0: it should not
    ## contain the 'name' and 'in' properties, and references to it should be from
    ## inside a schema.  However, Swagger parses it right this way, and doing it
    ## the official way shows the description only in the Models section, not inside
    ## the endpoint description, which is annoying
definitions:
  Trialname:
    name: trialname
    in: path
    type: string
    description: >
      Trial name provided by IPIN.

      Competitors will individually receive one ore more scoring trial names.  Scoring trials
      are not reloadable, meaning they can be used only once and can not be restarted when finished.
      IPIN will also provide some test trials for testing and tuning which are reloadable
      and can be reset to nonstarted state by using the `reload` request.

      The `test` and `demo` trials, both reloadable, are always available.  The `test` trial
      can be used for quickly testing the interface and provides arbitrary strings as data, while
      the `demo` trial provides data from Logfiles/01-Training/01a-Regular/T03_02.txt taken
      from http://indoorloc.uji.es/ipin2020track3/files/logfiles2020.zip"""
                                  )
try:
    from flasgger import Swagger, MK_SANITIZER

    Swagger(app, config=swagger_config, template=swagger_template, sanitizer=MK_SANITIZER)
except ImportError:
    pass


## This class is used inside a trial to provide logging services
class Triallog:
    sep = " ================ "  # predefined string for messages

    # trialname: name of the containing trial
    # logfn: name of the log file
    # logf: the stream object for the log file

    ## Constructor
    def __init__(self, trial):
        self.trialname = trial['name']  # name of this trial
        self.logfn = trial['logfn']
        openflag = 'a' if trial['reloadable'] else 'x'
        try:
            logf = lzma.open(self.logfn, openflag)  # log file descriptor
        except IOError:
            abort(Response("Cannot write log for " + self.trialname, 500))
        self.logf = logf

    ## Create date string
    def __date(self):
        ## ISO format with milliseconds: YYYY-MM-DDTHH:MM:SS.mmm
        return datetime.now(timezone.utc).strftime("%FT%H:%M:%S.%f")[0:-3]

    ## Low-level logger
    def __log(self, entrystr):
        if debug: print("logging entry: " + entrystr)
        try:
            self.logf.write(bytes(entrystr + '\n', 'ascii'))
        except IOError:
            abort(Response("Cannot log new entry for " + self.trialname, 500))

    ## CLose log file
    def close(self):
        if self.logf is not None:  # if file opened
            self.logf.close()

    ## Close and reopen log file to make sure it's flushed to disk
    def reopen(self):
        if self.logf is not None:  # if file opened
            self.logf.close()
            try:
                self.logf = lzma.open(self.logfn, 'a')
            except IOError:
                self.logf = None  # log file not opened
                abort(Response("Cannot reopen log for " + self.trialname, 500))

    ## Log functions
    def log_msg(self, msg):
        self.__log(self.__date() + self.sep + msg)

    def log_request(self):
        self.__log("%s <-- %s, %s%s" %
                   (self.__date(), request.remote_addr, request.host, request.full_path))

    def log_response(self, trial, response):
        self.__log("%s --> %d, trialts=%.3f rem=%.3f" %
                   (self.__date(), response.status_code, trial['trialts'], remaining_time(trial)))
        self.__log(response.data.decode('ascii'))


## Return the timestamp (first numeric field) of a data line
def get_line_timestamp(line, sep):
    fields = line.strip().split(sep)  # split on commas
    for field in fields:  # look at fields starting from first
        try:  # is this a numeric field?
            datats = float(field)  # try to interpret it as a timestamp
        except ValueError:  # not a numeric field
            continue  # try next one
        else:  # yes, we found it
            return datats
    return None  # no numeric field found


## Compute remaining time for a running trial
def remaining_time(trial):
    return round(trial['p'] + trial['V'] * trial['h'] + trial['s'] - time.time(), 3)


## Get the trial TRIALNAME from the trials dict, or create it from config files
def get_trial(trialname):
    ## If we don't know about trialname, read the init files and the test trial
    initrials = {}
    if trialname not in trials:
        trialinifn = trialsdir + trialname + ".yaml"
        if trialname in test_trial:
            if debug: conforigin = "internal test"
            initrials = test_trial
        elif os.path.isfile(trialinifn):
            if debug: conforigin = "file " + trialinifn
            with open(trialinifn, 'r') as inif:
                initrials = yaml.safe_load(inif)
        elif os.path.isfile(globinifn):
            if debug: conforigin = "file " + globinifn
            with open(globinifn, 'r') as inif:
                initrials = yaml.safe_load(inif)

        ## Create trials from initrials, set state
        for name in initrials:
            initrial = initrials[name]
            if name in trials:
                state = trials[name]['state']
                if state != 'nonstarted':
                    if debug: print(f"Trial {name} not reloaded from init file {globinifn} because it is {state}")
                    continue
            ## Create trial
            trials[name] = {'name': name,
                            'addr': set(),  # remote addresses accessing this trial
                            'logfn': trialsdir + name + ".log.xz",
                            'estfn': trialsdir + name + ".est",
                            'state': 'nonstarted'}
            try:
                for k in (  # selected keys from initrial
                        'datafile',  # text data file
                        'commsep',  # comment-starting string in data file
                        'sepch',  # field separator in data file
                        'V',  # trial time slowdown factor
                        'S',  # timeout slack time (seconds)
                        'inipos',  # initial position
                        'reloadable'  # not an scoring trial
                ):
                    trials[name][k] = initrial[k];  # copy from initrial
                assert isinstance(initrial['datafile'], str)
                assert isinstance(initrial['commsep'], str)
                assert isinstance(initrial['sepch'], str)
                assert isinstance(initrial['V'], (int, float)) and initrial['V'] >= 1
                assert isinstance(initrial['S'], (int, float)) and initrial['S'] >= 0
                assert isinstance(initrial['inipos'], str)
                assert isinstance(initrial['reloadable'], bool)
            except:
                abort(Response(f"Bad config for trial {name}", 500))
            if debug: print(f"Trial {name} loaded from {conforigin}")

    ## Return the trial, if it exists
    if trialname in trials:
        trial = trials[trialname]
        trial['addr'].add(request.remote_addr)
        return trial  # got and initialised it
    else:
        time.sleep(0.1)  # slow down brute-force attacks
        return None


## Close trial: dump estimates, close log and data file descriptors
def close_trial(trial):
    if 'est' in trial:
        try:
            with open(trial['estfn'], 'w') as estf:
                estf.write("\n".join(trial['est']))
        except IOError:
            abort(Response("Cannot write estimates", 500))
    if 'dataf' in trial:
        trial['dataf'].close()  # close the data file
        del trial['dataf']
    if 'log' in trial:
        trial['log'].close()  # close the Triallog
        del trial['log']


################################################################
#### Documentation endpoints
################################################################

htmlhead = """<!DOCTYPE html><html lang=en><head><meta charset="utf-8"/>
<title>{title}</title><link rel="icon" href="{favicon}"/></head><body>
""".format(title="EvaalAPI",
           favicon="https://evaal.aaloa.org/templates/evaal2012/favicon.ico")


## Make input strings safer
def safestr(string):
    return escape(string).striptags()


@app.route("/evaalapi/")
def root():
    ## Remove indentation from apidescription, least it be rendered literally
    lines = apidescription.splitlines()  # separate lines
    sl = [i.lstrip() for i in lines]  # remove leftmost spaces
    desc = '\n'.join(sl)  # join stripped lines

    return htmlhead + markdown("Revision " + revision + desc)


@app.route("/evaalapi/evaalapi.html")
def doc_html():
    with open(sourcedir + "evaalapi.md", 'r') as sf:
        return htmlhead + markdown(sf.read())


def html_beutify_sourcedir_file(filename, lexername, title=None):
    if title is None: title = filename
    lexer = get_lexer_by_name(lexername, stripall=True)
    formatter = HtmlFormatter(full=True, title=title, linenos='table', wrapcode=True)
    with open(sourcedir + filename, 'r') as sf:
        code = sf.read()
    return highlight(code, lexer, formatter)


@app.route("/evaalapi/evaalapi.md")
def doc_md():
    ## TODO: "md" → "markdown" once python3-pygments is upgraded to 2.8
    return html_beutify_sourcedir_file("evaalapi.md", "md")


@app.route("/evaalapi/" + source)
def python_source():
    return html_beutify_sourcedir_file(source, "python")


@app.route("/evaalapi/evaalapi.yaml")
def conf_source():
    return html_beutify_sourcedir_file("evaalapi.yaml", "yaml")


@app.route("/evaalapi/demo")
def demo_source():
    return html_beutify_sourcedir_file("demo", "python", "EvaalAPI demo")


@app.route("/evaalapi/demo-auto.out")
def demo_out():
    ## TODO: "text" → "output" once python3-pygments is upgraded to 2.10
    return html_beutify_sourcedir_file("demo-auto.out", "text")


################################################################
#### Management endpoints
################################################################

@app.route("/evaalapi/revision")
def server_revision():
    return {'start_time': starttime, 'revision': revision}
    """Return revision and start time of the server.

    ---
    produces: application/json
    responses:
      200:
        description: Returns server revision and start time.
        schema:
          type: string
          example: {"revision":"2.47","start_time":"2021-11-13T20:17:16"}

    """


@app.route("/evaalapi/status/<op>/<key>")
@app.route("/evaalapi/status/<key>", defaults={'op': 'complete'})
def trial_status(op, key):
    key = safestr(key)
    try:
        with open(privatefn, 'r') as inif:
            ini = yaml.safe_load(inif)
        masterkey = ini['masterkey']
    except (IOError, KeyError):
        return "Cannot get global status", 500

    if key != masterkey:
        time.sleep(1)  # slow down brute-force attacks
        return revision()

    ts = {}

    # ts[n] = {k: v for k, v in trials[n].items() if k not in ('log', 'dataf', 'response_data')}
    if op == 'complete':
        for trialname in trials:
            ts[trialname] = {}
            for k, v in trials[trialname].items():
                if k in ('log', 'dataf', 'response_data'):
                    pass
                elif k == 'addr':
                    ts[trialname][k] = list(v)
                elif k == 'est':
                    ts[trialname]['lastest'] = v[-1]
                elif k == 'line' and v is not None:
                    ts[trialname][k] = v.strip()
                else:
                    ts[trialname][k] = v

    elif op == 'state':
        ## Return state for each trial
        for trialname in trials:
            ts[trialname] = trials[trialname]['state']

    elif op == 'running':
        ## Return remaining time for running trials
        for trialname in trials:
            if trials[trialname]['state'] == 'running':
                ts[trialname] = remaining_time(trials[trialname])

    return ts


################################################################
#### EvaalAPI endpoints
################################################################

## Used at the beginning of route definitions for the EvaalAPI endpoints.
def check_request(trialname, valid_params):
    err = None
    trialname = safestr(trialname)

    ## Check for existence of trial
    trial = get_trial(trialname)
    if trial is None:
        err = ("Trial does not exist", 404)

    ## Check parameters
    invalid_params = set(request.args.keys()).difference(valid_params)
    if len(invalid_params) > 0:
        invalid_param = list(invalid_params)[0]
        err = (f"Invalid parameter {invalid_param} -- request ignored", 422)

    return trialname, trial, err


@app.route("/evaalapi/<trialname>/reload")
def trial_reload(trialname):
    """Reload trial data, set nonstarted state
    Returns trial state after reload, as with the `state` request in nonstarted state.

    Works for a testing trial or a nonstarted scoring trial.  Does not work for a scoring
    trial which has generated a log.

    ---
    externalDocs:
      description: See the complete documentation.
      url: /evaalapi/evaalapi.html
    parameters:
      - $ref: "#/definitions/Trialname"
      - name: keeplog
        in: query
        allowEmptyValue: true
        description: If present, keeps log and appends to it; if absent, deletes the log.
        required: false
    produces: text/csv
    responses:
      200:
        description: Returns the state of the nonstarted trial.
        schema:
          type: string
          example: 0.000,-1.000,3.000,15.000,0.000,0.000,0.000,10.422057;43.718278;1
      404:
        description: The trial does not exist.
      405:
        description: Trial is unstarted, no estimates are there.
      422:
        description: Invalid parameter or trial not reloadable.

    """
    valid_parameters = {'keeplog'}
    trialname, trial, error = check_request(trialname, valid_parameters)
    if error is not None: return error

    logfn = trial['logfn']
    estfn = trial['estfn']
    logfnexisted = os.path.exists(logfn)
    if debug: print(f"File {logfn} did{'' if logfnexisted else ' not'} exist")

    if 'log' in trial:  # trial not closed
        state = trial['state']
        assert state == 'running', state
        trial['log'].log_request()
        trial['log'].reopen()  # flush the compressed data
        logfnexists = os.path.exists(logfn)
    else:
        logfnexists = logfnexisted

    assert not (logfnexisted and not logfnexists)
    if debug: print(f"File {logfn} does{'' if logfnexists else ' not'} exist")

    if not trial['reloadable']:
        if logfnexisted:
            return (f"Trial {trialname} is not reloadable", 422)
        else:
            ## A non-reloadable trial can be reloaded if it has not yet produced any logs
            ## This opens a way to restart a non reloadable trial by first removing the log file
            if debug: print(f"Reloading non-reloadable trial {trialname}")

    close_trial(trial)
    del trials[trialname]
    if not logfnexisted or 'keeplog' not in request.args:
        if logfnexists: os.remove(logfn)
        if os.path.exists(estfn): os.remove(estfn)
    trial = get_trial(trialname)  # recreate trial from conf

    return trial_state(trialname, trial)


################################################################

@app.route("/evaalapi/<trialname>/state")
def trial_state(trialname, trial=None, dolog=True):
    """Get trial state
    Returns the trial state

    Useful to get the state of the trial for consistency checking and for getting
    the time remaining until timeout.
    ---
    externalDocs:
      description: See the complete documentation.
      url: /evaalapi/evaalapi.html
    parameters:
      - $ref: "#/definitions/Trialname"
    produces: text/csv
    responses:
      200:
        description: >
          Returns the state of the trial as a CSV line

          Returns a single data line in CSV format.  From Python, the returned data line can be parsed with
          `parse("{trialts:.3f},{rem:.3f},{V:.3f},{S:.3f},{p:.3f},{h:.3f},{pts:.3f},{pos}", line)`.
          See the complete documentation for the meaning of the returned values.
        schema:
          type: string
          example: 0.000,-1.000,3.000,15.000,0.000,0.000,0.000,10.422057;43.718278;1
      404:
        description: The trial does not exist.
      405:
        description: Trial is unstarted, no estimates are there.
      422:
        description: Invalid parameter or trial not reloadable.

    """
    if trial is None:  # called via app.route
        valid_parameters = {}
        trialname, trial, error = check_request(trialname, valid_parameters)
        if error is not None: return error

    state = trial['state']
    V = trial['V']
    S = trial['S']

    if dolog and 'log' in trial:  # trial nonstarted or not closed
        trial['log'].log_request()

    if state == 'nonstarted':
        inipos = trial['inipos']
        data = statefmt.format(trialts=0, rem=-1, V=V, S=S, p=0, h=0, pts=0, pos=inipos)
        if debug: print(f"State is {state}: {data}")
        ## Do not log this response, because trial is not started
        return data, {"Content-Type": "text/csv"}

    p = trial['p']
    h = trial['h']
    s = trial['s']
    est = trial['est'][-1]
    if debug: print(f"Last estimate is: \"{est}\"")
    lastest = parse(estfmt, est)  # parse last estimate
    assert lastest is not None
    pts = lastest['pts']  # timestamp of last estimate
    pos = lastest['pos']  # position of last estimate
    if state == 'running':
        trialts = trial['trialts']
        rem = remaining_time(trial)  # time remaining until timeout
    elif state == 'finished':
        trialts = -1
        rem = 0
    elif state == 'timeout':
        trialts = -1
        rem = s
    else:
        assert False, f"Unknown state '{state}'"

    data = statefmt.format(trialts=trialts, rem=rem, V=V, S=S, p=p, h=h, pts=pts, pos=pos)
    r = Response(data, 200, {"Content-Type": "text/csv"})

    if dolog and 'log' in trial:  # trial nonstarted or not closed
        trial['log'].log_response(trial, r)
    if debug: print(f"State is {state}: {data}")
    return r


################################################################

def trial_nextdata_response(trial):
    if debug:
        nlc = trial['response_data'].count('\n')
        print(f"Sending {nlc} lines of data")
    r = Response(trial['response_data'], 200, {"Content-Type": "text/csv"});
    trial['log'].log_response(trial, r)
    return r


@app.route("/evaalapi/<trialname>/nextdata")
def trial_nextdata(trialname):
    """Get data and set estimated position

    This is the main endpoint.
    For a scoring trial, this is the only necessary request to use.
    At each request, the competitor asks for next data, starting
    from the current trial timestamp
    through the requested `horizon` (default 0.5 seconds).  With the same request,
    the competitor can set a `position` estimate relative to the current timestamp.

    If the trial has not started yet, starts it and make it running.  If the trial is
    not finished, returns the data for the requested `horizon`.
    If the trial is finished, returns the status as with the `state` request.

    See the complete documentation for detail and timeout management.
    ---
    externalDocs:
      description: See the complete documentation.
      url: /evaalapi/evaalapi.html
    parameters:
      - $ref: "#/definitions/Trialname"
      - name: horizon
        in: query
        type: float
        allowEmptyValue: true
        description: A non-negative number of seconds.
        required: false
        default: 0.5
        minimum: 0
        maximum: 3e10
      - name: position
        in: query
        type: string
        description: Estimated position at current trial timestamp. Format is trial-dependent.
        required: false
        default: none
    produces: text/csv
    responses:
      200:
        description: Data lines in the requested time horizon.
        schema:
          type: string
          example: |
            ACCE;0.015;17287.341;-2.64559;8.69813;2.78446;3
            GYRO;0.016;17287.337;0.27428;-1.10933;-0.18998;3
            MAGN;0.016;17287.342;7.56000;-26.28000;-39.84000;3
            GYRO;0.016;17287.342;0.17776;-1.35429;-0.19364;3
            ACCE;0.017;17287.346;-2.82995;8.87052;2.36547;3
            AHRS;0.017;17287.346;51.891617;31.498898;-54.021839;0.48601067;0.02620481;-0.28725025;3
            ACCE;0.017;17287.351;-2.96402;9.06684;2.32716;3
      404:
        description: The trial does not exist.
      405:
        description: >
          Trial has finished, whether normally or because of a timeout;
          the position estimate is ignored; the trial timestamp does not change.
          Returns the trial state, as with the `state` request in nonstarted state.
        schema:
          type: string
          example: -1.000,0.000,1.000,2.000,1630239724.795,66.000,7.600,10.422059,43.718278,1
      422:
        description: >
          Invalid parameter or parameter value;
          the position estimate is ignored;
          the trial timestamp does not change; no data are returned.
      423:
        description: >
          This non-reloadable trial is configured with V > 2,
          meaning that the client is expected to be on average twice slower than real time.
          Yet, the time interval between the previous request and this one was smaller than
          the horizon of the previous request;
          the position estimate is ignored;
          the trial timestamp does not change; no data are returned.

    """
    valid_parameters = {'horizon', 'position', 'label'}
    trialname, trial, error = check_request(trialname, valid_parameters)
    if error is not None: return error

    ## Get state
    state = trial['state']
    assert state in states, state

    ## Log trial creation before logging request
    if state == 'nonstarted':
        log = trial['log'] = Triallog(trial)  # first, init log
        r = trial_state(trialname, trial, dolog=False)
        ## When nonstarted, trial_state returns a tuple rather than a Response
        ## so we do not use trial_state(trialname,trial)[0].data.decode('ascii')
        log.log_msg(f"Creating trial '{trialname}'\nstate: {trial_state(trialname, trial, dolog=False)[0]}")

    ## The Triallog is not there if the trial is closed
    if 'log' in trial:
        assert state in ('nonstarted', 'running'), state
        log = trial['log']
        log.log_request()
    else:
        assert state in ('finished', 'timeout'), state
        log = None

    ################################################################
    ## Return trial state if trial is finished
    if state in ('finished', 'timeout'):
        if debug: print(f"State is {state}")
        return trial_state(trialname, trial, dolog=False), 405

    ################################################################
    ## Start the trial if not started
    if state in ('nonstarted'):
        if debug: print(f"State is {state}")

        # Open data file
        datafn = trialsdir + trial['datafile']
        linets = None
        try:
            dataf = open(datafn, 'r')
            ## Skip initial headers, comments and empty lines
            while linets is None:  # a line without a numeric field
                line = next(dataf)
                if line.lstrip().startswith(trial['commsep']):
                    continue  # skip comments
                linets = get_line_timestamp(line, trial['sepch'])
        except (IOError, StopIteration):
            return "Error reading data file for " + trialname, 500

        ## Advance the state, set the remaining trial variables
        state = 'running'  # start the trial
        S = trial['S']
        inipos = trial['inipos']
        iniest = [esthdr,  # first line is header
                  estfmt.format(  # initial estimate
                      pts=linets,  # estimation timestamp
                      c=0,  # estimation wall clock
                      h=0,  # horizon requested at estimation
                      s=S,  # slack time at estimation
                      pos=inipos  # position estimate
                  )]
        trial.update({'state': state,  # running
                      'inits': linets,  # initial timestamp, from first trial timestamp
                      'trialts': linets,  # trial timestamp
                      'p': 0,  # wall clock of previous step
                      'h': 0,  # horizon of previous step
                      's': S,  # timeout slack (inited to S)
                      'line': line,  # first data line
                      'dataf': dataf,  # d-ata file descriptor
                      'est': iniest,  # initial estimate
                      'label': None  # command label for retry request
                      })

    ################################################################
    ## Running state: advance to next horizon
    assert state == 'running'
    if debug: print(f"State is {state}")
    V = trial['V']
    S = trial['S']
    trialts = trial['trialts']
    p = trial['p']
    h = trial['h']
    s = trial['s']
    dataf = trial['dataf']
    sepch = trial['sepch']
    line = trial['line']

    ## Check for timeout
    assert s >= 0 and s <= S
    c = time.time()
    if p > 0:  # p == 0 for first state request call
        ## V > 2 means that we expect the client to be twice slower than real time
        ## while (c-p) < h means that it is faster: ignore this request
        if not trial['reloadable'] and V > 2 and (c - p) < h:
            r = Response(f"Too fast, please slow down -- request ignored", 423)
            log.log_response(trial, r);
            return r
        s += V * h - (c - p)  # same as s=remaining_time(trial)
        if s > S:  # remaining greater than slack time
            s = S
        elif s < 0:  # no remaining time
            state = 'timeout'

    ## Check for finished trial
    if line is None:  # no more data lines
        state = 'finished'

    ## Unless timed out, read horizon and update the trial estimates
    if state in ('running', 'finished'):

        ## If using the same label as the previous request, serve the same data
        label = None
        if 'label' in request.args:
            label = safestr(request.args['label'])

            if label == trial['label']:
                ## This is a retry request: return same data as last requested
                if debug:
                    print(f"Nextadata retry request with label {label}")
                return trial_nextdata_response(trial)

        ## Read and validate horizon
        if 'horizon' in request.args:
            horizon = safestr(request.args['horizon'])
            try:
                h = float(horizon)
            except ValueError:
                r = Response(f"Horizon value not a number: \"{horizon}\" -- request ignored", 422)
                log.log_response(trial, r);
                return r
            if h < 0 or h == float('inf'):  # negative or infinite
                r = Response(f"Horizon negative or inf: \"{horizon}\" -- request ignored", 422)
                log.log_response(trial, r);
                return r
        else:
            h = 0.5  # default horizon (seconds)

        ## Update the trial estimates
        if (trialts > trial['inits']  # position at 0 is the initial position
                and 'position' in request.args):

            pos = safestr(request.args['position'])
            estimate = estfmt.format(
                pts=trialts,  # estimation timestamp
                c=c,  # estimation wall clock
                h=h,  # horizon requested at estimation
                s=s,  # slack time at estimation
                pos=pos  # position estimate
            )
            if debug: print(f"Estimate is: \"{estimate}\"")
            trial['est'].append(estimate)  # append to list of estimates

    ## If trial finished, stop here
    if state in ('finished', 'timeout'):
        trial['s'] = s  # save slack time
        trial['state'] = state  # set finished or timeout state
        r = trial_state(trialname, trial, dolog=False)
        r.status_code = 405;
        log.log_response(trial, r);
        log.log_msg(f"Trial finished {'normally' if state == 'finished' else 'by timeout'}\n")
        close_trial(trial)
        return r

    ## Get data for the requested horizon
    assert state == 'running'
    linets = get_line_timestamp(line, sepch)
    assert linets is not None
    assert linets >= trialts
    data = ""  # data to return
    datats = 0  # last data timestamp
    while linets < trialts + h:  # while line in horizon
        data += line  # append line to data
        datats = linets  # only used for consistency check
        try:
            line = next(dataf)  # read next line of data file
        except IOError:
            r = Response("IO error while reading data file for " + trialname, 500)
            log.log_response(trial, r);
            return r
        except StopIteration:  # no more lines in data file
            line = None
            break
        ## Get new line timestamp and check it
        linets = get_line_timestamp(line, sepch)
        if linets is None:
            r = Response("No timestamp in data file for " + trialname, 500)
            log.log_response(trial, r);
            return r
        if linets < datats:  # invalid timestamp
            r = Response("Decreasing timestamp in data file for " + trialname, 500)
            log.log_response(trial, r);
            return r

    ## Advance trial timestamp, update trial variables
    trial.update({'state': state,
                  'trialts': trialts + h,  # trial timestamp
                  'p': c,  # wall clock of last step
                  'h': h,  # horizon of last step
                  's': s,  # timeout slack
                  'line': line,  # first line beyond horizon
                  'response_data': data,  # data to send
                  'label': label  # command label for retry request
                  })

    ## Return data in the requested horizon
    return trial_nextdata_response(trial)


################################################################

@app.route("/evaalapi/<trialname>/estimates")
def trial_estimates(trialname):
    """Get estimates
    Returns a list of the estimates in CSV format with header.

    An estimate is an estimated position sent via the `nextdata` request preceded by
    timestamp and other relevant parameters.  Right after starting the trial, a
    header and the initial estimate are returned.  While running and after finished,
    all the estimates sent so far via the `nextdata` request are returned, one per
    data line.
    ---
    externalDocs:
      description: See the complete documentation.
      url: /evaalapi/evaalapi.html
    parameters:
      - $ref: "#/definitions/Trialname"
    produces: text/csv
    responses:
      200:
        description: >
          CSV list of estimates.

          The first line of data is a header.  Subsequent lines all have the same format.
          From Python, a data line after the header can be parsed with
          `parse("{pts:.3f},{c:.3f},{h:.3f},{s:.3f},{pos}", line)`.
          See the complete documentation for the meaning of the returned values.
        schema:
          type: string
          example: |
            pts,c,h,s,pos
            0.000,0.000,0.000,15.000,10.422057;43.718278;1
            1.000,1630165968.001,0.500,7.475,10.422059;43.718278;1
            1.500,1630165969.517,0.500,7.459,10.422063;43.718270;1
            2.000,1630165971.013,0.500,7.462,10.422071;43.718268;1
            2.500,1630165972.516,0.500,7.460,10.422072;43.718267;1
            3.000,1630165974.023,5.000,7.452,10.422069;43.718266;1
      404:
        description: The trial does not exist.
      405:
        description: Trial is unstarted, no estimates are there.
      422:
        description: Invalid parameter.

    """
    valid_parameters = {}
    trialname, trial, error = check_request(trialname, valid_parameters)
    if error is not None: return error

    ## If unstarted, no estimates exist
    if trial['state'] == 'nonstarted':
        return "Trial is not started", 405

    data = "\n".join(trial['est'])  # estimates separated by newlines
    estfn = trial['estfn']
    return data, {"Content-Type": "text/csv",
                  "Content-Disposition": "attachment; filename=" + estfn}


################################################################

@app.route("/evaalapi/<trialname>/log")
def trial_log(trialname):
    """Get the trial log
    Returns the log in text format of timestamped `state` and `nextdata`
    requests and responses.  `state` requests are logged only while the trial is running,
    that is after the first `nextdata` request and before it is finished.

    The log file exists only after the trial has started, which happens after the first
    `nextdata` request.  It is deleted after a successful `reload` request, unless used
    with the `keeplog` parameter.
    ---
    externalDocs:
      description: See the complete documentation.
      url: /evaalapi/evaalapi.html
    parameters:
      - $ref: "#/definitions/Trialname"
      - name: xzcompr
        in: query
        allowEmptyValue: true
        description: If present, log is returned in xz compressed form.
        required: false
    produces: text/plain or application/x-xz
    responses:
      200:
        description: The complete trial log.
        schema:
          type: string
          example: |
            2021-08-28T15:52:37.455 ================ Creating trial 'test'
            state: 0.000,-1.000,3.000,15.000,0.000,0.000,0.000,10.422057;43.718278;1
            2021-08-28T15:52:37.456 <-- evaal.aaloa.org/evaalapi/demo/nextdata?horizon=1
            2021-08-28T15:52:37.516 --> 200, trialts=1.000 rem=17.983
            ACCE;0.015;17287.341;-2.64559;8.69813;2.78446;3
            GYRO;0.016;17287.337;0.27428;-1.10933;-0.18998;3
            MAGN;0.016;17287.342;7.56000;-26.28000;-39.84000;3
      404:
        description: The trial does not exist.
      405:
        description: The trial log does not exist because the trial has not started.
      422:
        description: Invalid parameter.

    """
    valid_parameters = {'xzcompr'}
    trialname, trial, error = check_request(trialname, valid_parameters)
    if error is not None: return error

    logfn = trial['logfn']
    if not os.path.exists(logfn):
        return "Log file does not exist", 405

    if 'log' in trial:  # log file is still open
        if debug: print(f"Reopening log file {logfn}")
        trial['log'].reopen()  # flush the compressed data

    ## Send the log
    try:
        if 'xzcompr' in request.args:
            ## Send the log file as it is
            return send_file(logfn)
        else:
            ## Create a byte stream and close the file before returning
            # with lzma.open(logfn, 'r') as logf:
            #     stream = io.BytesIO(logf.read())
            ## Send the log decompressed the easy (and hopefully correct) way
            stream = lzma.open(logfn, 'r')
            return send_file(stream, mimetype="text/plain", as_attachment=True,
                             attachment_filename=trialname + ".log")
    except IOError:
        return "Cannot read log file for " + trialname, 500


## Main code
if os.path.isfile(guisource):
    with open(guisource, 'r') as guif:
        exec(compile(guif.read() + '\n', guisource, 'exec'))
    extrafiles = [guisource]
else:
    extrafiles = []

if __name__ == "__main__":
    app.run(extrafiles)

# Local Variables:
# comment-column: 40
# fill-column: 100
# End:
