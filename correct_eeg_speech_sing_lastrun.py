#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Fri Apr 26 11:56:08 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'eeg_speech_sing_2'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/evgeniataranova/onedrive/Fag/berkeley/knight/David_Alexis/antonyms/correct_eeg_speech_sing_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1440, 900], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='ptb')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='PsychToolbox')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    # Make folder to store recordings from micResponseSpeechRep
    micResponseSpeechRepRecFolder = filename + '_micResponseSpeechRep_recorded'
    if not os.path.isdir(micResponseSpeechRepRecFolder):
        os.mkdir(micResponseSpeechRepRecFolder)
    # Make folder to store recordings from micResponseSpeechGen
    micResponseSpeechGenRecFolder = filename + '_micResponseSpeechGen_recorded'
    if not os.path.isdir(micResponseSpeechGenRecFolder):
        os.mkdir(micResponseSpeechGenRecFolder)
    # Make folder to store recordings from micResponseSongRep
    micResponseSongRepRecFolder = filename + '_micResponseSongRep_recorded'
    if not os.path.isdir(micResponseSongRepRecFolder):
        os.mkdir(micResponseSongRepRecFolder)
    # Make folder to store recordings from micResponseSongGen
    micResponseSongGenRecFolder = filename + '_micResponseSongGen_recorded'
    if not os.path.isdir(micResponseSongGenRecFolder):
        os.mkdir(micResponseSongGenRecFolder)
    # Make folder to store recordings from micResponseSpeechRep_2
    micResponseSpeechRep_2RecFolder = filename + '_micResponseSpeechRep_2_recorded'
    if not os.path.isdir(micResponseSpeechRep_2RecFolder):
        os.mkdir(micResponseSpeechRep_2RecFolder)
    # Make folder to store recordings from micResponseSpeechGen_2
    micResponseSpeechGen_2RecFolder = filename + '_micResponseSpeechGen_2_recorded'
    if not os.path.isdir(micResponseSpeechGen_2RecFolder):
        os.mkdir(micResponseSpeechGen_2RecFolder)
    # Make folder to store recordings from micResponseSongRep_2
    micResponseSongRep_2RecFolder = filename + '_micResponseSongRep_2_recorded'
    if not os.path.isdir(micResponseSongRep_2RecFolder):
        os.mkdir(micResponseSongRep_2RecFolder)
    # Make folder to store recordings from micResponseSongGen_2
    micResponseSongGen_2RecFolder = filename + '_micResponseSongGen_2_recorded'
    if not os.path.isdir(micResponseSongGen_2RecFolder):
        os.mkdir(micResponseSongGen_2RecFolder)
    
    # --- Initialize components for Routine "Instruction_Start" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Instructions for the whole experiment\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Instruction_SpeechRep1" ---
    # create a microphone object for device: default
    defaultMicrophone = sound.microphone.Microphone(
        device=None, channels=None, 
        sampleRateHz=48000, maxRecordingSize=24000.0
    )
    
    # --- Initialize components for Routine "SpeechRepBlock1" ---
    wordPresSpeechRep = sound.Sound('A', secs=2, stereo=True, hamming=True,
        name='wordPresSpeechRep')
    wordPresSpeechRep.setVolume(1.0)
    # link micResponseSpeechRep to device object
    micResponseSpeechRep = defaultMicrophone
    buttonSpeechRep = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Rest" ---
    text = visual.TextStim(win=win, name='text',
        text='Thank you! \n\n30 sec rest\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Instruction_SpeechGen1" ---
    
    # --- Initialize components for Routine "SpeechGenBlock1" ---
    wordPresSpeechGen = sound.Sound(file, secs=2, stereo=True, hamming=True,
        name='wordPresSpeechGen')
    wordPresSpeechGen.setVolume(1.0)
    # link micResponseSpeechGen to device object
    micResponseSpeechGen = defaultMicrophone
    buttonSpeechGen = visual.ButtonStim(win, 
        text='Click here', font='Arvo',
        pos=(0, 0),
        letterHeight=0.05,
        size=(0.5, 0.5), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='buttonSpeechGen',
        depth=-2
    )
    buttonSpeechGen.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "Rest" ---
    text = visual.TextStim(win=win, name='text',
        text='Thank you! \n\n30 sec rest\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Instruction_SongRep1" ---
    
    # --- Initialize components for Routine "SongRepBlock1" ---
    wordPresSongRep = sound.Sound(file, secs=2, stereo=True, hamming=True,
        name='wordPresSongRep')
    wordPresSongRep.setVolume(1.0)
    # link micResponseSongRep to device object
    micResponseSongRep = defaultMicrophone
    buttonSongRep = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Rest" ---
    text = visual.TextStim(win=win, name='text',
        text='Thank you! \n\n30 sec rest\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Instruction_SongGen1" ---
    
    # --- Initialize components for Routine "SongGenBlock1" ---
    wordPresSongGen = sound.Sound(file, secs=2, stereo=True, hamming=True,
        name='wordPresSongGen')
    wordPresSongGen.setVolume(1.0)
    # link micResponseSongGen to device object
    micResponseSongGen = defaultMicrophone
    buttonSongGen = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Rest" ---
    text = visual.TextStim(win=win, name='text',
        text='Thank you! \n\n30 sec rest\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Instruction_SpeechRep2" ---
    
    # --- Initialize components for Routine "SpeechRepBlock2" ---
    wordPresSpeechRep_2 = sound.Sound(file, secs=2, stereo=True, hamming=True,
        name='wordPresSpeechRep_2')
    wordPresSpeechRep_2.setVolume(1.0)
    # link micResponseSpeechRep_2 to device object
    micResponseSpeechRep_2 = defaultMicrophone
    buttonSpeechRep_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Rest" ---
    text = visual.TextStim(win=win, name='text',
        text='Thank you! \n\n30 sec rest\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Instruction_SpeechGen2" ---
    
    # --- Initialize components for Routine "SpeechGenBlock2" ---
    wordPresSpeechGen_2 = sound.Sound(file, secs=2, stereo=True, hamming=True,
        name='wordPresSpeechGen_2')
    wordPresSpeechGen_2.setVolume(1.0)
    # link micResponseSpeechGen_2 to device object
    micResponseSpeechGen_2 = defaultMicrophone
    buttonSpeechGen_2 = visual.ButtonStim(win, 
        text='Click here', font='Arvo',
        pos=(0, 0),
        letterHeight=0.05,
        size=(0.5, 0.5), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='buttonSpeechGen_2',
        depth=-2
    )
    buttonSpeechGen_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "Rest" ---
    text = visual.TextStim(win=win, name='text',
        text='Thank you! \n\n30 sec rest\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Instruction_SongRep2" ---
    
    # --- Initialize components for Routine "SongRepBlock2" ---
    wordPresSongRep_2 = sound.Sound(file, secs=2, stereo=True, hamming=True,
        name='wordPresSongRep_2')
    wordPresSongRep_2.setVolume(1.0)
    # link micResponseSongRep_2 to device object
    micResponseSongRep_2 = defaultMicrophone
    buttonSongRep_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Rest" ---
    text = visual.TextStim(win=win, name='text',
        text='Thank you! \n\n30 sec rest\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Instruction_SongGen2" ---
    
    # --- Initialize components for Routine "SongGenBlock2" ---
    wordPresSongGen_2 = sound.Sound(file, secs=2, stereo=True, hamming=True,
        name='wordPresSongGen_2')
    wordPresSongGen_2.setVolume(1.0)
    # link micResponseSongGen_2 to device object
    micResponseSongGen_2 = defaultMicrophone
    buttonSongGen_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Finish" ---
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Instruction_Start" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instruction_Start.started', globalClock.getTime())
    # keep track of which components have finished
    Instruction_StartComponents = [text_2]
    for thisComponent in Instruction_StartComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruction_Start" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # if text_2 is stopping this frame...
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.stopped')
                # update status
                text_2.status = FINISHED
                text_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Instruction_StartComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruction_Start" ---
    for thisComponent in Instruction_StartComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instruction_Start.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    
    # set up handler to look after randomisation of conditions etc
    trials_1_block1 = data.TrialHandler(nReps=105.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('words_correct_speak_Repetition_Block1.csv'),
        seed=None, name='trials_1_block1')
    thisExp.addLoop(trials_1_block1)  # add the loop to the experiment
    thisTrials_1_block1 = trials_1_block1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_1_block1.rgb)
    if thisTrials_1_block1 != None:
        for paramName in thisTrials_1_block1:
            globals()[paramName] = thisTrials_1_block1[paramName]
    
    for thisTrials_1_block1 in trials_1_block1:
        currentLoop = trials_1_block1
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_1_block1.rgb)
        if thisTrials_1_block1 != None:
            for paramName in thisTrials_1_block1:
                globals()[paramName] = thisTrials_1_block1[paramName]
        
        # --- Prepare to start Routine "Instruction_SpeechRep1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instruction_SpeechRep1.started', globalClock.getTime())
        # keep track of which components have finished
        Instruction_SpeechRep1Components = []
        for thisComponent in Instruction_SpeechRep1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instruction_SpeechRep1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instruction_SpeechRep1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instruction_SpeechRep1" ---
        for thisComponent in Instruction_SpeechRep1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instruction_SpeechRep1.stopped', globalClock.getTime())
        # the Routine "Instruction_SpeechRep1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "SpeechRepBlock1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('SpeechRepBlock1.started', globalClock.getTime())
        wordPresSpeechRep.setSound(file, secs=2, hamming=True)
        wordPresSpeechRep.setVolume(1.0, log=False)
        wordPresSpeechRep.seek(0)
        buttonSpeechRep.keys = []
        buttonSpeechRep.rt = []
        _buttonSpeechRep_allKeys = []
        # keep track of which components have finished
        SpeechRepBlock1Components = [wordPresSpeechRep, micResponseSpeechRep, buttonSpeechRep]
        for thisComponent in SpeechRepBlock1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SpeechRepBlock1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # if wordPresSpeechRep is starting this frame...
            if wordPresSpeechRep.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                wordPresSpeechRep.frameNStart = frameN  # exact frame index
                wordPresSpeechRep.tStart = t  # local t and not account for scr refresh
                wordPresSpeechRep.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('wordPresSpeechRep.started', tThisFlipGlobal)
                # update status
                wordPresSpeechRep.status = STARTED
                wordPresSpeechRep.play(when=win)  # sync with win flip
            
            # if wordPresSpeechRep is stopping this frame...
            if wordPresSpeechRep.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > wordPresSpeechRep.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    wordPresSpeechRep.tStop = t  # not accounting for scr refresh
                    wordPresSpeechRep.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wordPresSpeechRep.stopped')
                    # update status
                    wordPresSpeechRep.status = FINISHED
                    wordPresSpeechRep.stop()
            # update wordPresSpeechRep status according to whether it's playing
            if wordPresSpeechRep.isPlaying:
                wordPresSpeechRep.status = STARTED
            elif wordPresSpeechRep.isFinished:
                wordPresSpeechRep.status = FINISHED
            
            # if micResponseSpeechRep is starting this frame...
            if micResponseSpeechRep.status == NOT_STARTED and t >= 1-frameTolerance:
                # keep track of start time/frame for later
                micResponseSpeechRep.frameNStart = frameN  # exact frame index
                micResponseSpeechRep.tStart = t  # local t and not account for scr refresh
                micResponseSpeechRep.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(micResponseSpeechRep, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('micResponseSpeechRep.started', t)
                # update status
                micResponseSpeechRep.status = STARTED
                # start recording with micResponseSpeechRep
                micResponseSpeechRep.start()
            
            # if micResponseSpeechRep is active this frame...
            if micResponseSpeechRep.status == STARTED:
                # update params
                pass
                # update recorded clip for micResponseSpeechRep
                micResponseSpeechRep.poll()
            
            # if micResponseSpeechRep is stopping this frame...
            if micResponseSpeechRep.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > micResponseSpeechRep.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    micResponseSpeechRep.tStop = t  # not accounting for scr refresh
                    micResponseSpeechRep.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('micResponseSpeechRep.stopped', t)
                    # update status
                    micResponseSpeechRep.status = FINISHED
                    # stop recording with micResponseSpeechRep
                    micResponseSpeechRep.stop()
            
            # *buttonSpeechRep* updates
            waitOnFlip = False
            
            # if buttonSpeechRep is starting this frame...
            if buttonSpeechRep.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
                # keep track of start time/frame for later
                buttonSpeechRep.frameNStart = frameN  # exact frame index
                buttonSpeechRep.tStart = t  # local t and not account for scr refresh
                buttonSpeechRep.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(buttonSpeechRep, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'buttonSpeechRep.started')
                # update status
                buttonSpeechRep.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(buttonSpeechRep.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(buttonSpeechRep.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if buttonSpeechRep is stopping this frame...
            if buttonSpeechRep.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > buttonSpeechRep.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    buttonSpeechRep.tStop = t  # not accounting for scr refresh
                    buttonSpeechRep.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonSpeechRep.stopped')
                    # update status
                    buttonSpeechRep.status = FINISHED
                    buttonSpeechRep.status = FINISHED
            if buttonSpeechRep.status == STARTED and not waitOnFlip:
                theseKeys = buttonSpeechRep.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _buttonSpeechRep_allKeys.extend(theseKeys)
                if len(_buttonSpeechRep_allKeys):
                    buttonSpeechRep.keys = _buttonSpeechRep_allKeys[-1].name  # just the last key pressed
                    buttonSpeechRep.rt = _buttonSpeechRep_allKeys[-1].rt
                    buttonSpeechRep.duration = _buttonSpeechRep_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SpeechRepBlock1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SpeechRepBlock1" ---
        for thisComponent in SpeechRepBlock1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('SpeechRepBlock1.stopped', globalClock.getTime())
        wordPresSpeechRep.pause()  # ensure sound has stopped at end of Routine
        # tell mic to keep hold of current recording in micResponseSpeechRep.clips and transcript (if applicable) in micResponseSpeechRep.scripts
        # this will also update micResponseSpeechRep.lastClip and micResponseSpeechRep.lastScript
        micResponseSpeechRep.stop()
        tag = data.utils.getDateStr()
        micResponseSpeechRepClip = micResponseSpeechRep.bank(
            tag=tag, transcribe='None',
            config=None
        )
        trials_1_block1.addData('micResponseSpeechRep.clip', os.path.join(micResponseSpeechRepRecFolder, 'recording_micResponseSpeechRep_%s.wav' % tag))
        # check responses
        if buttonSpeechRep.keys in ['', [], None]:  # No response was made
            buttonSpeechRep.keys = None
        trials_1_block1.addData('buttonSpeechRep.keys',buttonSpeechRep.keys)
        if buttonSpeechRep.keys != None:  # we had a response
            trials_1_block1.addData('buttonSpeechRep.rt', buttonSpeechRep.rt)
            trials_1_block1.addData('buttonSpeechRep.duration', buttonSpeechRep.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 105.0 repeats of 'trials_1_block1'
    
    
    # --- Prepare to start Routine "Rest" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Rest.started', globalClock.getTime())
    # keep track of which components have finished
    RestComponents = [text]
    for thisComponent in RestComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Rest" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 30.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RestComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Rest" ---
    for thisComponent in RestComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Rest.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-30.000000)
    
    # set up handler to look after randomisation of conditions etc
    trials_2_block1 = data.TrialHandler(nReps=105.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('words_correct_speak_Generation_Block1.csv'),
        seed=None, name='trials_2_block1')
    thisExp.addLoop(trials_2_block1)  # add the loop to the experiment
    thisTrials_2_block1 = trials_2_block1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_2_block1.rgb)
    if thisTrials_2_block1 != None:
        for paramName in thisTrials_2_block1:
            globals()[paramName] = thisTrials_2_block1[paramName]
    
    for thisTrials_2_block1 in trials_2_block1:
        currentLoop = trials_2_block1
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_2_block1.rgb)
        if thisTrials_2_block1 != None:
            for paramName in thisTrials_2_block1:
                globals()[paramName] = thisTrials_2_block1[paramName]
        
        # --- Prepare to start Routine "Instruction_SpeechGen1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instruction_SpeechGen1.started', globalClock.getTime())
        # keep track of which components have finished
        Instruction_SpeechGen1Components = []
        for thisComponent in Instruction_SpeechGen1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instruction_SpeechGen1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instruction_SpeechGen1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instruction_SpeechGen1" ---
        for thisComponent in Instruction_SpeechGen1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instruction_SpeechGen1.stopped', globalClock.getTime())
        # the Routine "Instruction_SpeechGen1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "SpeechGenBlock1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('SpeechGenBlock1.started', globalClock.getTime())
        wordPresSpeechGen.setSound(file, secs=2, hamming=True)
        wordPresSpeechGen.setVolume(1.0, log=False)
        wordPresSpeechGen.seek(0)
        # reset buttonSpeechGen to account for continued clicks & clear times on/off
        buttonSpeechGen.reset()
        # keep track of which components have finished
        SpeechGenBlock1Components = [wordPresSpeechGen, micResponseSpeechGen, buttonSpeechGen]
        for thisComponent in SpeechGenBlock1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SpeechGenBlock1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # if wordPresSpeechGen is starting this frame...
            if wordPresSpeechGen.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                wordPresSpeechGen.frameNStart = frameN  # exact frame index
                wordPresSpeechGen.tStart = t  # local t and not account for scr refresh
                wordPresSpeechGen.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('wordPresSpeechGen.started', tThisFlipGlobal)
                # update status
                wordPresSpeechGen.status = STARTED
                wordPresSpeechGen.play(when=win)  # sync with win flip
            
            # if wordPresSpeechGen is stopping this frame...
            if wordPresSpeechGen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > wordPresSpeechGen.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    wordPresSpeechGen.tStop = t  # not accounting for scr refresh
                    wordPresSpeechGen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wordPresSpeechGen.stopped')
                    # update status
                    wordPresSpeechGen.status = FINISHED
                    wordPresSpeechGen.stop()
            # update wordPresSpeechGen status according to whether it's playing
            if wordPresSpeechGen.isPlaying:
                wordPresSpeechGen.status = STARTED
            elif wordPresSpeechGen.isFinished:
                wordPresSpeechGen.status = FINISHED
            
            # if micResponseSpeechGen is starting this frame...
            if micResponseSpeechGen.status == NOT_STARTED and t >= 1-frameTolerance:
                # keep track of start time/frame for later
                micResponseSpeechGen.frameNStart = frameN  # exact frame index
                micResponseSpeechGen.tStart = t  # local t and not account for scr refresh
                micResponseSpeechGen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(micResponseSpeechGen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('micResponseSpeechGen.started', t)
                # update status
                micResponseSpeechGen.status = STARTED
                # start recording with micResponseSpeechGen
                micResponseSpeechGen.start()
            
            # if micResponseSpeechGen is active this frame...
            if micResponseSpeechGen.status == STARTED:
                # update params
                pass
                # update recorded clip for micResponseSpeechGen
                micResponseSpeechGen.poll()
            
            # if micResponseSpeechGen is stopping this frame...
            if micResponseSpeechGen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > micResponseSpeechGen.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    micResponseSpeechGen.tStop = t  # not accounting for scr refresh
                    micResponseSpeechGen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('micResponseSpeechGen.stopped', t)
                    # update status
                    micResponseSpeechGen.status = FINISHED
                    # stop recording with micResponseSpeechGen
                    micResponseSpeechGen.stop()
            # *buttonSpeechGen* updates
            
            # if buttonSpeechGen is starting this frame...
            if buttonSpeechGen.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
                # keep track of start time/frame for later
                buttonSpeechGen.frameNStart = frameN  # exact frame index
                buttonSpeechGen.tStart = t  # local t and not account for scr refresh
                buttonSpeechGen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(buttonSpeechGen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'buttonSpeechGen.started')
                # update status
                buttonSpeechGen.status = STARTED
                buttonSpeechGen.setAutoDraw(True)
            
            # if buttonSpeechGen is active this frame...
            if buttonSpeechGen.status == STARTED:
                # update params
                pass
                # check whether buttonSpeechGen has been pressed
                if buttonSpeechGen.isClicked:
                    if not buttonSpeechGen.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        buttonSpeechGen.timesOn.append(buttonSpeechGen.buttonClock.getTime())
                        buttonSpeechGen.timesOff.append(buttonSpeechGen.buttonClock.getTime())
                    elif len(buttonSpeechGen.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        buttonSpeechGen.timesOff[-1] = buttonSpeechGen.buttonClock.getTime()
                    if not buttonSpeechGen.wasClicked:
                        # end routine when buttonSpeechGen is clicked
                        continueRoutine = False
                    if not buttonSpeechGen.wasClicked:
                        # run callback code when buttonSpeechGen is clicked
                        pass
            # take note of whether buttonSpeechGen was clicked, so that next frame we know if clicks are new
            buttonSpeechGen.wasClicked = buttonSpeechGen.isClicked and buttonSpeechGen.status == STARTED
            
            # if buttonSpeechGen is stopping this frame...
            if buttonSpeechGen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > buttonSpeechGen.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    buttonSpeechGen.tStop = t  # not accounting for scr refresh
                    buttonSpeechGen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonSpeechGen.stopped')
                    # update status
                    buttonSpeechGen.status = FINISHED
                    buttonSpeechGen.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SpeechGenBlock1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SpeechGenBlock1" ---
        for thisComponent in SpeechGenBlock1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('SpeechGenBlock1.stopped', globalClock.getTime())
        wordPresSpeechGen.pause()  # ensure sound has stopped at end of Routine
        # tell mic to keep hold of current recording in micResponseSpeechGen.clips and transcript (if applicable) in micResponseSpeechGen.scripts
        # this will also update micResponseSpeechGen.lastClip and micResponseSpeechGen.lastScript
        micResponseSpeechGen.stop()
        tag = data.utils.getDateStr()
        micResponseSpeechGenClip = micResponseSpeechGen.bank(
            tag=tag, transcribe='None',
            config=None
        )
        trials_2_block1.addData('micResponseSpeechGen.clip', os.path.join(micResponseSpeechGenRecFolder, 'recording_micResponseSpeechGen_%s.wav' % tag))
        trials_2_block1.addData('buttonSpeechGen.numClicks', buttonSpeechGen.numClicks)
        if buttonSpeechGen.numClicks:
           trials_2_block1.addData('buttonSpeechGen.timesOn', buttonSpeechGen.timesOn)
           trials_2_block1.addData('buttonSpeechGen.timesOff', buttonSpeechGen.timesOff)
        else:
           trials_2_block1.addData('buttonSpeechGen.timesOn', "")
           trials_2_block1.addData('buttonSpeechGen.timesOff', "")
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 105.0 repeats of 'trials_2_block1'
    
    
    # --- Prepare to start Routine "Rest" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Rest.started', globalClock.getTime())
    # keep track of which components have finished
    RestComponents = [text]
    for thisComponent in RestComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Rest" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 30.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RestComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Rest" ---
    for thisComponent in RestComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Rest.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-30.000000)
    
    # set up handler to look after randomisation of conditions etc
    trials_3_block1 = data.TrialHandler(nReps=105.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('words_correct_sing_Repetition_Block1.csv'),
        seed=None, name='trials_3_block1')
    thisExp.addLoop(trials_3_block1)  # add the loop to the experiment
    thisTrials_3_block1 = trials_3_block1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_3_block1.rgb)
    if thisTrials_3_block1 != None:
        for paramName in thisTrials_3_block1:
            globals()[paramName] = thisTrials_3_block1[paramName]
    
    for thisTrials_3_block1 in trials_3_block1:
        currentLoop = trials_3_block1
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_3_block1.rgb)
        if thisTrials_3_block1 != None:
            for paramName in thisTrials_3_block1:
                globals()[paramName] = thisTrials_3_block1[paramName]
        
        # --- Prepare to start Routine "Instruction_SongRep1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instruction_SongRep1.started', globalClock.getTime())
        # keep track of which components have finished
        Instruction_SongRep1Components = []
        for thisComponent in Instruction_SongRep1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instruction_SongRep1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instruction_SongRep1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instruction_SongRep1" ---
        for thisComponent in Instruction_SongRep1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instruction_SongRep1.stopped', globalClock.getTime())
        # the Routine "Instruction_SongRep1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "SongRepBlock1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('SongRepBlock1.started', globalClock.getTime())
        wordPresSongRep.setSound(file, secs=2, hamming=True)
        wordPresSongRep.setVolume(1.0, log=False)
        wordPresSongRep.seek(0)
        buttonSongRep.keys = []
        buttonSongRep.rt = []
        _buttonSongRep_allKeys = []
        # keep track of which components have finished
        SongRepBlock1Components = [wordPresSongRep, micResponseSongRep, buttonSongRep]
        for thisComponent in SongRepBlock1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SongRepBlock1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # if wordPresSongRep is starting this frame...
            if wordPresSongRep.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                wordPresSongRep.frameNStart = frameN  # exact frame index
                wordPresSongRep.tStart = t  # local t and not account for scr refresh
                wordPresSongRep.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('wordPresSongRep.started', tThisFlipGlobal)
                # update status
                wordPresSongRep.status = STARTED
                wordPresSongRep.play(when=win)  # sync with win flip
            
            # if wordPresSongRep is stopping this frame...
            if wordPresSongRep.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > wordPresSongRep.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    wordPresSongRep.tStop = t  # not accounting for scr refresh
                    wordPresSongRep.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wordPresSongRep.stopped')
                    # update status
                    wordPresSongRep.status = FINISHED
                    wordPresSongRep.stop()
            # update wordPresSongRep status according to whether it's playing
            if wordPresSongRep.isPlaying:
                wordPresSongRep.status = STARTED
            elif wordPresSongRep.isFinished:
                wordPresSongRep.status = FINISHED
            
            # if micResponseSongRep is starting this frame...
            if micResponseSongRep.status == NOT_STARTED and t >= 1-frameTolerance:
                # keep track of start time/frame for later
                micResponseSongRep.frameNStart = frameN  # exact frame index
                micResponseSongRep.tStart = t  # local t and not account for scr refresh
                micResponseSongRep.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(micResponseSongRep, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('micResponseSongRep.started', t)
                # update status
                micResponseSongRep.status = STARTED
                # start recording with micResponseSongRep
                micResponseSongRep.start()
            
            # if micResponseSongRep is active this frame...
            if micResponseSongRep.status == STARTED:
                # update params
                pass
                # update recorded clip for micResponseSongRep
                micResponseSongRep.poll()
            
            # if micResponseSongRep is stopping this frame...
            if micResponseSongRep.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > micResponseSongRep.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    micResponseSongRep.tStop = t  # not accounting for scr refresh
                    micResponseSongRep.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('micResponseSongRep.stopped', t)
                    # update status
                    micResponseSongRep.status = FINISHED
                    # stop recording with micResponseSongRep
                    micResponseSongRep.stop()
            
            # *buttonSongRep* updates
            waitOnFlip = False
            
            # if buttonSongRep is starting this frame...
            if buttonSongRep.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
                # keep track of start time/frame for later
                buttonSongRep.frameNStart = frameN  # exact frame index
                buttonSongRep.tStart = t  # local t and not account for scr refresh
                buttonSongRep.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(buttonSongRep, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'buttonSongRep.started')
                # update status
                buttonSongRep.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(buttonSongRep.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(buttonSongRep.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if buttonSongRep is stopping this frame...
            if buttonSongRep.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > buttonSongRep.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    buttonSongRep.tStop = t  # not accounting for scr refresh
                    buttonSongRep.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonSongRep.stopped')
                    # update status
                    buttonSongRep.status = FINISHED
                    buttonSongRep.status = FINISHED
            if buttonSongRep.status == STARTED and not waitOnFlip:
                theseKeys = buttonSongRep.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _buttonSongRep_allKeys.extend(theseKeys)
                if len(_buttonSongRep_allKeys):
                    buttonSongRep.keys = _buttonSongRep_allKeys[-1].name  # just the last key pressed
                    buttonSongRep.rt = _buttonSongRep_allKeys[-1].rt
                    buttonSongRep.duration = _buttonSongRep_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SongRepBlock1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SongRepBlock1" ---
        for thisComponent in SongRepBlock1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('SongRepBlock1.stopped', globalClock.getTime())
        wordPresSongRep.pause()  # ensure sound has stopped at end of Routine
        # tell mic to keep hold of current recording in micResponseSongRep.clips and transcript (if applicable) in micResponseSongRep.scripts
        # this will also update micResponseSongRep.lastClip and micResponseSongRep.lastScript
        micResponseSongRep.stop()
        tag = data.utils.getDateStr()
        micResponseSongRepClip = micResponseSongRep.bank(
            tag=tag, transcribe='None',
            config=None
        )
        trials_3_block1.addData('micResponseSongRep.clip', os.path.join(micResponseSongRepRecFolder, 'recording_micResponseSongRep_%s.wav' % tag))
        # check responses
        if buttonSongRep.keys in ['', [], None]:  # No response was made
            buttonSongRep.keys = None
        trials_3_block1.addData('buttonSongRep.keys',buttonSongRep.keys)
        if buttonSongRep.keys != None:  # we had a response
            trials_3_block1.addData('buttonSongRep.rt', buttonSongRep.rt)
            trials_3_block1.addData('buttonSongRep.duration', buttonSongRep.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 105.0 repeats of 'trials_3_block1'
    
    
    # --- Prepare to start Routine "Rest" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Rest.started', globalClock.getTime())
    # keep track of which components have finished
    RestComponents = [text]
    for thisComponent in RestComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Rest" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 30.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RestComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Rest" ---
    for thisComponent in RestComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Rest.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-30.000000)
    
    # set up handler to look after randomisation of conditions etc
    trials_4_block1 = data.TrialHandler(nReps=105.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('words_correct_sing_Generation_Block1.csv'),
        seed=None, name='trials_4_block1')
    thisExp.addLoop(trials_4_block1)  # add the loop to the experiment
    thisTrials_4_block1 = trials_4_block1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_4_block1.rgb)
    if thisTrials_4_block1 != None:
        for paramName in thisTrials_4_block1:
            globals()[paramName] = thisTrials_4_block1[paramName]
    
    for thisTrials_4_block1 in trials_4_block1:
        currentLoop = trials_4_block1
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_4_block1.rgb)
        if thisTrials_4_block1 != None:
            for paramName in thisTrials_4_block1:
                globals()[paramName] = thisTrials_4_block1[paramName]
        
        # --- Prepare to start Routine "Instruction_SongGen1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instruction_SongGen1.started', globalClock.getTime())
        # keep track of which components have finished
        Instruction_SongGen1Components = []
        for thisComponent in Instruction_SongGen1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instruction_SongGen1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instruction_SongGen1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instruction_SongGen1" ---
        for thisComponent in Instruction_SongGen1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instruction_SongGen1.stopped', globalClock.getTime())
        # the Routine "Instruction_SongGen1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "SongGenBlock1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('SongGenBlock1.started', globalClock.getTime())
        wordPresSongGen.setSound(file, secs=2, hamming=True)
        wordPresSongGen.setVolume(1.0, log=False)
        wordPresSongGen.seek(0)
        buttonSongGen.keys = []
        buttonSongGen.rt = []
        _buttonSongGen_allKeys = []
        # keep track of which components have finished
        SongGenBlock1Components = [wordPresSongGen, micResponseSongGen, buttonSongGen]
        for thisComponent in SongGenBlock1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SongGenBlock1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # if wordPresSongGen is starting this frame...
            if wordPresSongGen.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                wordPresSongGen.frameNStart = frameN  # exact frame index
                wordPresSongGen.tStart = t  # local t and not account for scr refresh
                wordPresSongGen.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('wordPresSongGen.started', tThisFlipGlobal)
                # update status
                wordPresSongGen.status = STARTED
                wordPresSongGen.play(when=win)  # sync with win flip
            
            # if wordPresSongGen is stopping this frame...
            if wordPresSongGen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > wordPresSongGen.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    wordPresSongGen.tStop = t  # not accounting for scr refresh
                    wordPresSongGen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wordPresSongGen.stopped')
                    # update status
                    wordPresSongGen.status = FINISHED
                    wordPresSongGen.stop()
            # update wordPresSongGen status according to whether it's playing
            if wordPresSongGen.isPlaying:
                wordPresSongGen.status = STARTED
            elif wordPresSongGen.isFinished:
                wordPresSongGen.status = FINISHED
            
            # if micResponseSongGen is starting this frame...
            if micResponseSongGen.status == NOT_STARTED and t >= 1-frameTolerance:
                # keep track of start time/frame for later
                micResponseSongGen.frameNStart = frameN  # exact frame index
                micResponseSongGen.tStart = t  # local t and not account for scr refresh
                micResponseSongGen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(micResponseSongGen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('micResponseSongGen.started', t)
                # update status
                micResponseSongGen.status = STARTED
                # start recording with micResponseSongGen
                micResponseSongGen.start()
            
            # if micResponseSongGen is active this frame...
            if micResponseSongGen.status == STARTED:
                # update params
                pass
                # update recorded clip for micResponseSongGen
                micResponseSongGen.poll()
            
            # if micResponseSongGen is stopping this frame...
            if micResponseSongGen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > micResponseSongGen.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    micResponseSongGen.tStop = t  # not accounting for scr refresh
                    micResponseSongGen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('micResponseSongGen.stopped', t)
                    # update status
                    micResponseSongGen.status = FINISHED
                    # stop recording with micResponseSongGen
                    micResponseSongGen.stop()
            
            # *buttonSongGen* updates
            waitOnFlip = False
            
            # if buttonSongGen is starting this frame...
            if buttonSongGen.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
                # keep track of start time/frame for later
                buttonSongGen.frameNStart = frameN  # exact frame index
                buttonSongGen.tStart = t  # local t and not account for scr refresh
                buttonSongGen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(buttonSongGen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'buttonSongGen.started')
                # update status
                buttonSongGen.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(buttonSongGen.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(buttonSongGen.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if buttonSongGen is stopping this frame...
            if buttonSongGen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > buttonSongGen.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    buttonSongGen.tStop = t  # not accounting for scr refresh
                    buttonSongGen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonSongGen.stopped')
                    # update status
                    buttonSongGen.status = FINISHED
                    buttonSongGen.status = FINISHED
            if buttonSongGen.status == STARTED and not waitOnFlip:
                theseKeys = buttonSongGen.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _buttonSongGen_allKeys.extend(theseKeys)
                if len(_buttonSongGen_allKeys):
                    buttonSongGen.keys = _buttonSongGen_allKeys[-1].name  # just the last key pressed
                    buttonSongGen.rt = _buttonSongGen_allKeys[-1].rt
                    buttonSongGen.duration = _buttonSongGen_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SongGenBlock1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SongGenBlock1" ---
        for thisComponent in SongGenBlock1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('SongGenBlock1.stopped', globalClock.getTime())
        wordPresSongGen.pause()  # ensure sound has stopped at end of Routine
        # tell mic to keep hold of current recording in micResponseSongGen.clips and transcript (if applicable) in micResponseSongGen.scripts
        # this will also update micResponseSongGen.lastClip and micResponseSongGen.lastScript
        micResponseSongGen.stop()
        tag = data.utils.getDateStr()
        micResponseSongGenClip = micResponseSongGen.bank(
            tag=tag, transcribe='None',
            config=None
        )
        trials_4_block1.addData('micResponseSongGen.clip', os.path.join(micResponseSongGenRecFolder, 'recording_micResponseSongGen_%s.wav' % tag))
        # check responses
        if buttonSongGen.keys in ['', [], None]:  # No response was made
            buttonSongGen.keys = None
        trials_4_block1.addData('buttonSongGen.keys',buttonSongGen.keys)
        if buttonSongGen.keys != None:  # we had a response
            trials_4_block1.addData('buttonSongGen.rt', buttonSongGen.rt)
            trials_4_block1.addData('buttonSongGen.duration', buttonSongGen.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 105.0 repeats of 'trials_4_block1'
    
    
    # --- Prepare to start Routine "Rest" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Rest.started', globalClock.getTime())
    # keep track of which components have finished
    RestComponents = [text]
    for thisComponent in RestComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Rest" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 30.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RestComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Rest" ---
    for thisComponent in RestComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Rest.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-30.000000)
    
    # set up handler to look after randomisation of conditions etc
    trials_1_block2 = data.TrialHandler(nReps=105.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('words_correct_speak_Repetition_Block2.csv'),
        seed=None, name='trials_1_block2')
    thisExp.addLoop(trials_1_block2)  # add the loop to the experiment
    thisTrials_1_block2 = trials_1_block2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_1_block2.rgb)
    if thisTrials_1_block2 != None:
        for paramName in thisTrials_1_block2:
            globals()[paramName] = thisTrials_1_block2[paramName]
    
    for thisTrials_1_block2 in trials_1_block2:
        currentLoop = trials_1_block2
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_1_block2.rgb)
        if thisTrials_1_block2 != None:
            for paramName in thisTrials_1_block2:
                globals()[paramName] = thisTrials_1_block2[paramName]
        
        # --- Prepare to start Routine "Instruction_SpeechRep2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instruction_SpeechRep2.started', globalClock.getTime())
        # keep track of which components have finished
        Instruction_SpeechRep2Components = []
        for thisComponent in Instruction_SpeechRep2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instruction_SpeechRep2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instruction_SpeechRep2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instruction_SpeechRep2" ---
        for thisComponent in Instruction_SpeechRep2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instruction_SpeechRep2.stopped', globalClock.getTime())
        # the Routine "Instruction_SpeechRep2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "SpeechRepBlock2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('SpeechRepBlock2.started', globalClock.getTime())
        wordPresSpeechRep_2.setSound(file, secs=2, hamming=True)
        wordPresSpeechRep_2.setVolume(1.0, log=False)
        wordPresSpeechRep_2.seek(0)
        buttonSpeechRep_2.keys = []
        buttonSpeechRep_2.rt = []
        _buttonSpeechRep_2_allKeys = []
        # keep track of which components have finished
        SpeechRepBlock2Components = [wordPresSpeechRep_2, micResponseSpeechRep_2, buttonSpeechRep_2]
        for thisComponent in SpeechRepBlock2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SpeechRepBlock2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # if wordPresSpeechRep_2 is starting this frame...
            if wordPresSpeechRep_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                wordPresSpeechRep_2.frameNStart = frameN  # exact frame index
                wordPresSpeechRep_2.tStart = t  # local t and not account for scr refresh
                wordPresSpeechRep_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('wordPresSpeechRep_2.started', tThisFlipGlobal)
                # update status
                wordPresSpeechRep_2.status = STARTED
                wordPresSpeechRep_2.play(when=win)  # sync with win flip
            
            # if wordPresSpeechRep_2 is stopping this frame...
            if wordPresSpeechRep_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > wordPresSpeechRep_2.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    wordPresSpeechRep_2.tStop = t  # not accounting for scr refresh
                    wordPresSpeechRep_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wordPresSpeechRep_2.stopped')
                    # update status
                    wordPresSpeechRep_2.status = FINISHED
                    wordPresSpeechRep_2.stop()
            # update wordPresSpeechRep_2 status according to whether it's playing
            if wordPresSpeechRep_2.isPlaying:
                wordPresSpeechRep_2.status = STARTED
            elif wordPresSpeechRep_2.isFinished:
                wordPresSpeechRep_2.status = FINISHED
            
            # if micResponseSpeechRep_2 is starting this frame...
            if micResponseSpeechRep_2.status == NOT_STARTED and t >= 1-frameTolerance:
                # keep track of start time/frame for later
                micResponseSpeechRep_2.frameNStart = frameN  # exact frame index
                micResponseSpeechRep_2.tStart = t  # local t and not account for scr refresh
                micResponseSpeechRep_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(micResponseSpeechRep_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('micResponseSpeechRep_2.started', t)
                # update status
                micResponseSpeechRep_2.status = STARTED
                # start recording with micResponseSpeechRep_2
                micResponseSpeechRep_2.start()
            
            # if micResponseSpeechRep_2 is active this frame...
            if micResponseSpeechRep_2.status == STARTED:
                # update params
                pass
                # update recorded clip for micResponseSpeechRep_2
                micResponseSpeechRep_2.poll()
            
            # if micResponseSpeechRep_2 is stopping this frame...
            if micResponseSpeechRep_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > micResponseSpeechRep_2.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    micResponseSpeechRep_2.tStop = t  # not accounting for scr refresh
                    micResponseSpeechRep_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('micResponseSpeechRep_2.stopped', t)
                    # update status
                    micResponseSpeechRep_2.status = FINISHED
                    # stop recording with micResponseSpeechRep_2
                    micResponseSpeechRep_2.stop()
            
            # *buttonSpeechRep_2* updates
            waitOnFlip = False
            
            # if buttonSpeechRep_2 is starting this frame...
            if buttonSpeechRep_2.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
                # keep track of start time/frame for later
                buttonSpeechRep_2.frameNStart = frameN  # exact frame index
                buttonSpeechRep_2.tStart = t  # local t and not account for scr refresh
                buttonSpeechRep_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(buttonSpeechRep_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'buttonSpeechRep_2.started')
                # update status
                buttonSpeechRep_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(buttonSpeechRep_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(buttonSpeechRep_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if buttonSpeechRep_2 is stopping this frame...
            if buttonSpeechRep_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > buttonSpeechRep_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    buttonSpeechRep_2.tStop = t  # not accounting for scr refresh
                    buttonSpeechRep_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonSpeechRep_2.stopped')
                    # update status
                    buttonSpeechRep_2.status = FINISHED
                    buttonSpeechRep_2.status = FINISHED
            if buttonSpeechRep_2.status == STARTED and not waitOnFlip:
                theseKeys = buttonSpeechRep_2.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _buttonSpeechRep_2_allKeys.extend(theseKeys)
                if len(_buttonSpeechRep_2_allKeys):
                    buttonSpeechRep_2.keys = _buttonSpeechRep_2_allKeys[-1].name  # just the last key pressed
                    buttonSpeechRep_2.rt = _buttonSpeechRep_2_allKeys[-1].rt
                    buttonSpeechRep_2.duration = _buttonSpeechRep_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SpeechRepBlock2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SpeechRepBlock2" ---
        for thisComponent in SpeechRepBlock2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('SpeechRepBlock2.stopped', globalClock.getTime())
        wordPresSpeechRep_2.pause()  # ensure sound has stopped at end of Routine
        # tell mic to keep hold of current recording in micResponseSpeechRep_2.clips and transcript (if applicable) in micResponseSpeechRep_2.scripts
        # this will also update micResponseSpeechRep_2.lastClip and micResponseSpeechRep_2.lastScript
        micResponseSpeechRep_2.stop()
        tag = data.utils.getDateStr()
        micResponseSpeechRep_2Clip = micResponseSpeechRep_2.bank(
            tag=tag, transcribe='None',
            config=None
        )
        trials_1_block2.addData('micResponseSpeechRep_2.clip', os.path.join(micResponseSpeechRep_2RecFolder, 'recording_micResponseSpeechRep_2_%s.wav' % tag))
        # check responses
        if buttonSpeechRep_2.keys in ['', [], None]:  # No response was made
            buttonSpeechRep_2.keys = None
        trials_1_block2.addData('buttonSpeechRep_2.keys',buttonSpeechRep_2.keys)
        if buttonSpeechRep_2.keys != None:  # we had a response
            trials_1_block2.addData('buttonSpeechRep_2.rt', buttonSpeechRep_2.rt)
            trials_1_block2.addData('buttonSpeechRep_2.duration', buttonSpeechRep_2.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 105.0 repeats of 'trials_1_block2'
    
    
    # --- Prepare to start Routine "Rest" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Rest.started', globalClock.getTime())
    # keep track of which components have finished
    RestComponents = [text]
    for thisComponent in RestComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Rest" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 30.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RestComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Rest" ---
    for thisComponent in RestComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Rest.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-30.000000)
    
    # set up handler to look after randomisation of conditions etc
    trials_2_block2 = data.TrialHandler(nReps=105.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('words_correct_speak_Generation_Block2.csv'),
        seed=None, name='trials_2_block2')
    thisExp.addLoop(trials_2_block2)  # add the loop to the experiment
    thisTrials_2_block2 = trials_2_block2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_2_block2.rgb)
    if thisTrials_2_block2 != None:
        for paramName in thisTrials_2_block2:
            globals()[paramName] = thisTrials_2_block2[paramName]
    
    for thisTrials_2_block2 in trials_2_block2:
        currentLoop = trials_2_block2
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_2_block2.rgb)
        if thisTrials_2_block2 != None:
            for paramName in thisTrials_2_block2:
                globals()[paramName] = thisTrials_2_block2[paramName]
        
        # --- Prepare to start Routine "Instruction_SpeechGen2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instruction_SpeechGen2.started', globalClock.getTime())
        # keep track of which components have finished
        Instruction_SpeechGen2Components = []
        for thisComponent in Instruction_SpeechGen2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instruction_SpeechGen2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instruction_SpeechGen2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instruction_SpeechGen2" ---
        for thisComponent in Instruction_SpeechGen2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instruction_SpeechGen2.stopped', globalClock.getTime())
        # the Routine "Instruction_SpeechGen2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "SpeechGenBlock2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('SpeechGenBlock2.started', globalClock.getTime())
        wordPresSpeechGen_2.setSound(file, secs=2, hamming=True)
        wordPresSpeechGen_2.setVolume(1.0, log=False)
        wordPresSpeechGen_2.seek(0)
        # reset buttonSpeechGen_2 to account for continued clicks & clear times on/off
        buttonSpeechGen_2.reset()
        # keep track of which components have finished
        SpeechGenBlock2Components = [wordPresSpeechGen_2, micResponseSpeechGen_2, buttonSpeechGen_2]
        for thisComponent in SpeechGenBlock2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SpeechGenBlock2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # if wordPresSpeechGen_2 is starting this frame...
            if wordPresSpeechGen_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                wordPresSpeechGen_2.frameNStart = frameN  # exact frame index
                wordPresSpeechGen_2.tStart = t  # local t and not account for scr refresh
                wordPresSpeechGen_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('wordPresSpeechGen_2.started', tThisFlipGlobal)
                # update status
                wordPresSpeechGen_2.status = STARTED
                wordPresSpeechGen_2.play(when=win)  # sync with win flip
            
            # if wordPresSpeechGen_2 is stopping this frame...
            if wordPresSpeechGen_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > wordPresSpeechGen_2.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    wordPresSpeechGen_2.tStop = t  # not accounting for scr refresh
                    wordPresSpeechGen_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wordPresSpeechGen_2.stopped')
                    # update status
                    wordPresSpeechGen_2.status = FINISHED
                    wordPresSpeechGen_2.stop()
            # update wordPresSpeechGen_2 status according to whether it's playing
            if wordPresSpeechGen_2.isPlaying:
                wordPresSpeechGen_2.status = STARTED
            elif wordPresSpeechGen_2.isFinished:
                wordPresSpeechGen_2.status = FINISHED
            
            # if micResponseSpeechGen_2 is starting this frame...
            if micResponseSpeechGen_2.status == NOT_STARTED and t >= 1-frameTolerance:
                # keep track of start time/frame for later
                micResponseSpeechGen_2.frameNStart = frameN  # exact frame index
                micResponseSpeechGen_2.tStart = t  # local t and not account for scr refresh
                micResponseSpeechGen_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(micResponseSpeechGen_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('micResponseSpeechGen_2.started', t)
                # update status
                micResponseSpeechGen_2.status = STARTED
                # start recording with micResponseSpeechGen_2
                micResponseSpeechGen_2.start()
            
            # if micResponseSpeechGen_2 is active this frame...
            if micResponseSpeechGen_2.status == STARTED:
                # update params
                pass
                # update recorded clip for micResponseSpeechGen_2
                micResponseSpeechGen_2.poll()
            
            # if micResponseSpeechGen_2 is stopping this frame...
            if micResponseSpeechGen_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > micResponseSpeechGen_2.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    micResponseSpeechGen_2.tStop = t  # not accounting for scr refresh
                    micResponseSpeechGen_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('micResponseSpeechGen_2.stopped', t)
                    # update status
                    micResponseSpeechGen_2.status = FINISHED
                    # stop recording with micResponseSpeechGen_2
                    micResponseSpeechGen_2.stop()
            # *buttonSpeechGen_2* updates
            
            # if buttonSpeechGen_2 is starting this frame...
            if buttonSpeechGen_2.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
                # keep track of start time/frame for later
                buttonSpeechGen_2.frameNStart = frameN  # exact frame index
                buttonSpeechGen_2.tStart = t  # local t and not account for scr refresh
                buttonSpeechGen_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(buttonSpeechGen_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'buttonSpeechGen_2.started')
                # update status
                buttonSpeechGen_2.status = STARTED
                buttonSpeechGen_2.setAutoDraw(True)
            
            # if buttonSpeechGen_2 is active this frame...
            if buttonSpeechGen_2.status == STARTED:
                # update params
                pass
                # check whether buttonSpeechGen_2 has been pressed
                if buttonSpeechGen_2.isClicked:
                    if not buttonSpeechGen_2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        buttonSpeechGen_2.timesOn.append(buttonSpeechGen_2.buttonClock.getTime())
                        buttonSpeechGen_2.timesOff.append(buttonSpeechGen_2.buttonClock.getTime())
                    elif len(buttonSpeechGen_2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        buttonSpeechGen_2.timesOff[-1] = buttonSpeechGen_2.buttonClock.getTime()
                    if not buttonSpeechGen_2.wasClicked:
                        # end routine when buttonSpeechGen_2 is clicked
                        continueRoutine = False
                    if not buttonSpeechGen_2.wasClicked:
                        # run callback code when buttonSpeechGen_2 is clicked
                        pass
            # take note of whether buttonSpeechGen_2 was clicked, so that next frame we know if clicks are new
            buttonSpeechGen_2.wasClicked = buttonSpeechGen_2.isClicked and buttonSpeechGen_2.status == STARTED
            
            # if buttonSpeechGen_2 is stopping this frame...
            if buttonSpeechGen_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > buttonSpeechGen_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    buttonSpeechGen_2.tStop = t  # not accounting for scr refresh
                    buttonSpeechGen_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonSpeechGen_2.stopped')
                    # update status
                    buttonSpeechGen_2.status = FINISHED
                    buttonSpeechGen_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SpeechGenBlock2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SpeechGenBlock2" ---
        for thisComponent in SpeechGenBlock2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('SpeechGenBlock2.stopped', globalClock.getTime())
        wordPresSpeechGen_2.pause()  # ensure sound has stopped at end of Routine
        # tell mic to keep hold of current recording in micResponseSpeechGen_2.clips and transcript (if applicable) in micResponseSpeechGen_2.scripts
        # this will also update micResponseSpeechGen_2.lastClip and micResponseSpeechGen_2.lastScript
        micResponseSpeechGen_2.stop()
        tag = data.utils.getDateStr()
        micResponseSpeechGen_2Clip = micResponseSpeechGen_2.bank(
            tag=tag, transcribe='None',
            config=None
        )
        trials_2_block2.addData('micResponseSpeechGen_2.clip', os.path.join(micResponseSpeechGen_2RecFolder, 'recording_micResponseSpeechGen_2_%s.wav' % tag))
        trials_2_block2.addData('buttonSpeechGen_2.numClicks', buttonSpeechGen_2.numClicks)
        if buttonSpeechGen_2.numClicks:
           trials_2_block2.addData('buttonSpeechGen_2.timesOn', buttonSpeechGen_2.timesOn)
           trials_2_block2.addData('buttonSpeechGen_2.timesOff', buttonSpeechGen_2.timesOff)
        else:
           trials_2_block2.addData('buttonSpeechGen_2.timesOn', "")
           trials_2_block2.addData('buttonSpeechGen_2.timesOff', "")
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 105.0 repeats of 'trials_2_block2'
    
    
    # --- Prepare to start Routine "Rest" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Rest.started', globalClock.getTime())
    # keep track of which components have finished
    RestComponents = [text]
    for thisComponent in RestComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Rest" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 30.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RestComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Rest" ---
    for thisComponent in RestComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Rest.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-30.000000)
    
    # set up handler to look after randomisation of conditions etc
    trials_3_block2 = data.TrialHandler(nReps=105.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('words_correct_sing_Repetition_Block2.csv'),
        seed=None, name='trials_3_block2')
    thisExp.addLoop(trials_3_block2)  # add the loop to the experiment
    thisTrials_3_block2 = trials_3_block2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_3_block2.rgb)
    if thisTrials_3_block2 != None:
        for paramName in thisTrials_3_block2:
            globals()[paramName] = thisTrials_3_block2[paramName]
    
    for thisTrials_3_block2 in trials_3_block2:
        currentLoop = trials_3_block2
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_3_block2.rgb)
        if thisTrials_3_block2 != None:
            for paramName in thisTrials_3_block2:
                globals()[paramName] = thisTrials_3_block2[paramName]
        
        # --- Prepare to start Routine "Instruction_SongRep2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instruction_SongRep2.started', globalClock.getTime())
        # keep track of which components have finished
        Instruction_SongRep2Components = []
        for thisComponent in Instruction_SongRep2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instruction_SongRep2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instruction_SongRep2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instruction_SongRep2" ---
        for thisComponent in Instruction_SongRep2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instruction_SongRep2.stopped', globalClock.getTime())
        # the Routine "Instruction_SongRep2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "SongRepBlock2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('SongRepBlock2.started', globalClock.getTime())
        wordPresSongRep_2.setSound(file, secs=2, hamming=True)
        wordPresSongRep_2.setVolume(1.0, log=False)
        wordPresSongRep_2.seek(0)
        buttonSongRep_2.keys = []
        buttonSongRep_2.rt = []
        _buttonSongRep_2_allKeys = []
        # keep track of which components have finished
        SongRepBlock2Components = [wordPresSongRep_2, micResponseSongRep_2, buttonSongRep_2]
        for thisComponent in SongRepBlock2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SongRepBlock2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # if wordPresSongRep_2 is starting this frame...
            if wordPresSongRep_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                wordPresSongRep_2.frameNStart = frameN  # exact frame index
                wordPresSongRep_2.tStart = t  # local t and not account for scr refresh
                wordPresSongRep_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('wordPresSongRep_2.started', tThisFlipGlobal)
                # update status
                wordPresSongRep_2.status = STARTED
                wordPresSongRep_2.play(when=win)  # sync with win flip
            
            # if wordPresSongRep_2 is stopping this frame...
            if wordPresSongRep_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > wordPresSongRep_2.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    wordPresSongRep_2.tStop = t  # not accounting for scr refresh
                    wordPresSongRep_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wordPresSongRep_2.stopped')
                    # update status
                    wordPresSongRep_2.status = FINISHED
                    wordPresSongRep_2.stop()
            # update wordPresSongRep_2 status according to whether it's playing
            if wordPresSongRep_2.isPlaying:
                wordPresSongRep_2.status = STARTED
            elif wordPresSongRep_2.isFinished:
                wordPresSongRep_2.status = FINISHED
            
            # if micResponseSongRep_2 is starting this frame...
            if micResponseSongRep_2.status == NOT_STARTED and t >= 1-frameTolerance:
                # keep track of start time/frame for later
                micResponseSongRep_2.frameNStart = frameN  # exact frame index
                micResponseSongRep_2.tStart = t  # local t and not account for scr refresh
                micResponseSongRep_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(micResponseSongRep_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('micResponseSongRep_2.started', t)
                # update status
                micResponseSongRep_2.status = STARTED
                # start recording with micResponseSongRep_2
                micResponseSongRep_2.start()
            
            # if micResponseSongRep_2 is active this frame...
            if micResponseSongRep_2.status == STARTED:
                # update params
                pass
                # update recorded clip for micResponseSongRep_2
                micResponseSongRep_2.poll()
            
            # if micResponseSongRep_2 is stopping this frame...
            if micResponseSongRep_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > micResponseSongRep_2.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    micResponseSongRep_2.tStop = t  # not accounting for scr refresh
                    micResponseSongRep_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('micResponseSongRep_2.stopped', t)
                    # update status
                    micResponseSongRep_2.status = FINISHED
                    # stop recording with micResponseSongRep_2
                    micResponseSongRep_2.stop()
            
            # *buttonSongRep_2* updates
            waitOnFlip = False
            
            # if buttonSongRep_2 is starting this frame...
            if buttonSongRep_2.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
                # keep track of start time/frame for later
                buttonSongRep_2.frameNStart = frameN  # exact frame index
                buttonSongRep_2.tStart = t  # local t and not account for scr refresh
                buttonSongRep_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(buttonSongRep_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'buttonSongRep_2.started')
                # update status
                buttonSongRep_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(buttonSongRep_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(buttonSongRep_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if buttonSongRep_2 is stopping this frame...
            if buttonSongRep_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > buttonSongRep_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    buttonSongRep_2.tStop = t  # not accounting for scr refresh
                    buttonSongRep_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonSongRep_2.stopped')
                    # update status
                    buttonSongRep_2.status = FINISHED
                    buttonSongRep_2.status = FINISHED
            if buttonSongRep_2.status == STARTED and not waitOnFlip:
                theseKeys = buttonSongRep_2.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _buttonSongRep_2_allKeys.extend(theseKeys)
                if len(_buttonSongRep_2_allKeys):
                    buttonSongRep_2.keys = _buttonSongRep_2_allKeys[-1].name  # just the last key pressed
                    buttonSongRep_2.rt = _buttonSongRep_2_allKeys[-1].rt
                    buttonSongRep_2.duration = _buttonSongRep_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SongRepBlock2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SongRepBlock2" ---
        for thisComponent in SongRepBlock2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('SongRepBlock2.stopped', globalClock.getTime())
        wordPresSongRep_2.pause()  # ensure sound has stopped at end of Routine
        # tell mic to keep hold of current recording in micResponseSongRep_2.clips and transcript (if applicable) in micResponseSongRep_2.scripts
        # this will also update micResponseSongRep_2.lastClip and micResponseSongRep_2.lastScript
        micResponseSongRep_2.stop()
        tag = data.utils.getDateStr()
        micResponseSongRep_2Clip = micResponseSongRep_2.bank(
            tag=tag, transcribe='None',
            config=None
        )
        trials_3_block2.addData('micResponseSongRep_2.clip', os.path.join(micResponseSongRep_2RecFolder, 'recording_micResponseSongRep_2_%s.wav' % tag))
        # check responses
        if buttonSongRep_2.keys in ['', [], None]:  # No response was made
            buttonSongRep_2.keys = None
        trials_3_block2.addData('buttonSongRep_2.keys',buttonSongRep_2.keys)
        if buttonSongRep_2.keys != None:  # we had a response
            trials_3_block2.addData('buttonSongRep_2.rt', buttonSongRep_2.rt)
            trials_3_block2.addData('buttonSongRep_2.duration', buttonSongRep_2.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 105.0 repeats of 'trials_3_block2'
    
    
    # --- Prepare to start Routine "Rest" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Rest.started', globalClock.getTime())
    # keep track of which components have finished
    RestComponents = [text]
    for thisComponent in RestComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Rest" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 30.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in RestComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Rest" ---
    for thisComponent in RestComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Rest.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-30.000000)
    
    # set up handler to look after randomisation of conditions etc
    trials_4_block2 = data.TrialHandler(nReps=105.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('words_correct_sing_Generation_Block2.csv'),
        seed=None, name='trials_4_block2')
    thisExp.addLoop(trials_4_block2)  # add the loop to the experiment
    thisTrials_4_block2 = trials_4_block2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_4_block2.rgb)
    if thisTrials_4_block2 != None:
        for paramName in thisTrials_4_block2:
            globals()[paramName] = thisTrials_4_block2[paramName]
    
    for thisTrials_4_block2 in trials_4_block2:
        currentLoop = trials_4_block2
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_4_block2.rgb)
        if thisTrials_4_block2 != None:
            for paramName in thisTrials_4_block2:
                globals()[paramName] = thisTrials_4_block2[paramName]
        
        # --- Prepare to start Routine "Instruction_SongGen2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instruction_SongGen2.started', globalClock.getTime())
        # keep track of which components have finished
        Instruction_SongGen2Components = []
        for thisComponent in Instruction_SongGen2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instruction_SongGen2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instruction_SongGen2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instruction_SongGen2" ---
        for thisComponent in Instruction_SongGen2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instruction_SongGen2.stopped', globalClock.getTime())
        # the Routine "Instruction_SongGen2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "SongGenBlock2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('SongGenBlock2.started', globalClock.getTime())
        wordPresSongGen_2.setSound(file, secs=2, hamming=True)
        wordPresSongGen_2.setVolume(1.0, log=False)
        wordPresSongGen_2.seek(0)
        buttonSongGen_2.keys = []
        buttonSongGen_2.rt = []
        _buttonSongGen_2_allKeys = []
        # keep track of which components have finished
        SongGenBlock2Components = [wordPresSongGen_2, micResponseSongGen_2, buttonSongGen_2]
        for thisComponent in SongGenBlock2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SongGenBlock2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # if wordPresSongGen_2 is starting this frame...
            if wordPresSongGen_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                wordPresSongGen_2.frameNStart = frameN  # exact frame index
                wordPresSongGen_2.tStart = t  # local t and not account for scr refresh
                wordPresSongGen_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('wordPresSongGen_2.started', tThisFlipGlobal)
                # update status
                wordPresSongGen_2.status = STARTED
                wordPresSongGen_2.play(when=win)  # sync with win flip
            
            # if wordPresSongGen_2 is stopping this frame...
            if wordPresSongGen_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > wordPresSongGen_2.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    wordPresSongGen_2.tStop = t  # not accounting for scr refresh
                    wordPresSongGen_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wordPresSongGen_2.stopped')
                    # update status
                    wordPresSongGen_2.status = FINISHED
                    wordPresSongGen_2.stop()
            # update wordPresSongGen_2 status according to whether it's playing
            if wordPresSongGen_2.isPlaying:
                wordPresSongGen_2.status = STARTED
            elif wordPresSongGen_2.isFinished:
                wordPresSongGen_2.status = FINISHED
            
            # if micResponseSongGen_2 is starting this frame...
            if micResponseSongGen_2.status == NOT_STARTED and t >= 1-frameTolerance:
                # keep track of start time/frame for later
                micResponseSongGen_2.frameNStart = frameN  # exact frame index
                micResponseSongGen_2.tStart = t  # local t and not account for scr refresh
                micResponseSongGen_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(micResponseSongGen_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('micResponseSongGen_2.started', t)
                # update status
                micResponseSongGen_2.status = STARTED
                # start recording with micResponseSongGen_2
                micResponseSongGen_2.start()
            
            # if micResponseSongGen_2 is active this frame...
            if micResponseSongGen_2.status == STARTED:
                # update params
                pass
                # update recorded clip for micResponseSongGen_2
                micResponseSongGen_2.poll()
            
            # if micResponseSongGen_2 is stopping this frame...
            if micResponseSongGen_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > micResponseSongGen_2.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    micResponseSongGen_2.tStop = t  # not accounting for scr refresh
                    micResponseSongGen_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('micResponseSongGen_2.stopped', t)
                    # update status
                    micResponseSongGen_2.status = FINISHED
                    # stop recording with micResponseSongGen_2
                    micResponseSongGen_2.stop()
            
            # *buttonSongGen_2* updates
            waitOnFlip = False
            
            # if buttonSongGen_2 is starting this frame...
            if buttonSongGen_2.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
                # keep track of start time/frame for later
                buttonSongGen_2.frameNStart = frameN  # exact frame index
                buttonSongGen_2.tStart = t  # local t and not account for scr refresh
                buttonSongGen_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(buttonSongGen_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'buttonSongGen_2.started')
                # update status
                buttonSongGen_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(buttonSongGen_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(buttonSongGen_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if buttonSongGen_2 is stopping this frame...
            if buttonSongGen_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > buttonSongGen_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    buttonSongGen_2.tStop = t  # not accounting for scr refresh
                    buttonSongGen_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonSongGen_2.stopped')
                    # update status
                    buttonSongGen_2.status = FINISHED
                    buttonSongGen_2.status = FINISHED
            if buttonSongGen_2.status == STARTED and not waitOnFlip:
                theseKeys = buttonSongGen_2.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _buttonSongGen_2_allKeys.extend(theseKeys)
                if len(_buttonSongGen_2_allKeys):
                    buttonSongGen_2.keys = _buttonSongGen_2_allKeys[-1].name  # just the last key pressed
                    buttonSongGen_2.rt = _buttonSongGen_2_allKeys[-1].rt
                    buttonSongGen_2.duration = _buttonSongGen_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SongGenBlock2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SongGenBlock2" ---
        for thisComponent in SongGenBlock2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('SongGenBlock2.stopped', globalClock.getTime())
        wordPresSongGen_2.pause()  # ensure sound has stopped at end of Routine
        # tell mic to keep hold of current recording in micResponseSongGen_2.clips and transcript (if applicable) in micResponseSongGen_2.scripts
        # this will also update micResponseSongGen_2.lastClip and micResponseSongGen_2.lastScript
        micResponseSongGen_2.stop()
        tag = data.utils.getDateStr()
        micResponseSongGen_2Clip = micResponseSongGen_2.bank(
            tag=tag, transcribe='None',
            config=None
        )
        trials_4_block2.addData('micResponseSongGen_2.clip', os.path.join(micResponseSongGen_2RecFolder, 'recording_micResponseSongGen_2_%s.wav' % tag))
        # check responses
        if buttonSongGen_2.keys in ['', [], None]:  # No response was made
            buttonSongGen_2.keys = None
        trials_4_block2.addData('buttonSongGen_2.keys',buttonSongGen_2.keys)
        if buttonSongGen_2.keys != None:  # we had a response
            trials_4_block2.addData('buttonSongGen_2.rt', buttonSongGen_2.rt)
            trials_4_block2.addData('buttonSongGen_2.duration', buttonSongGen_2.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 105.0 repeats of 'trials_4_block2'
    
    
    # --- Prepare to start Routine "Finish" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Finish.started', globalClock.getTime())
    # keep track of which components have finished
    FinishComponents = []
    for thisComponent in FinishComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Finish" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in FinishComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Finish" ---
    for thisComponent in FinishComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Finish.stopped', globalClock.getTime())
    # the Routine "Finish" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # save micResponseSpeechRep recordings
    for tag in micResponseSpeechRep.clips:
        for i, clip in enumerate(micResponseSpeechRep.clips[tag]):
            clipFilename = 'recording_micResponseSpeechRep_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micResponseSpeechRepRecFolder, clipFilename))
    # save micResponseSpeechGen recordings
    for tag in micResponseSpeechGen.clips:
        for i, clip in enumerate(micResponseSpeechGen.clips[tag]):
            clipFilename = 'recording_micResponseSpeechGen_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micResponseSpeechGenRecFolder, clipFilename))
    # save micResponseSongRep recordings
    for tag in micResponseSongRep.clips:
        for i, clip in enumerate(micResponseSongRep.clips[tag]):
            clipFilename = 'recording_micResponseSongRep_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micResponseSongRepRecFolder, clipFilename))
    # save micResponseSongGen recordings
    for tag in micResponseSongGen.clips:
        for i, clip in enumerate(micResponseSongGen.clips[tag]):
            clipFilename = 'recording_micResponseSongGen_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micResponseSongGenRecFolder, clipFilename))
    # save micResponseSpeechRep_2 recordings
    for tag in micResponseSpeechRep_2.clips:
        for i, clip in enumerate(micResponseSpeechRep_2.clips[tag]):
            clipFilename = 'recording_micResponseSpeechRep_2_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micResponseSpeechRep_2RecFolder, clipFilename))
    # save micResponseSpeechGen_2 recordings
    for tag in micResponseSpeechGen_2.clips:
        for i, clip in enumerate(micResponseSpeechGen_2.clips[tag]):
            clipFilename = 'recording_micResponseSpeechGen_2_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micResponseSpeechGen_2RecFolder, clipFilename))
    # save micResponseSongRep_2 recordings
    for tag in micResponseSongRep_2.clips:
        for i, clip in enumerate(micResponseSongRep_2.clips[tag]):
            clipFilename = 'recording_micResponseSongRep_2_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micResponseSongRep_2RecFolder, clipFilename))
    # save micResponseSongGen_2 recordings
    for tag in micResponseSongGen_2.clips:
        for i, clip in enumerate(micResponseSongGen_2.clips[tag]):
            clipFilename = 'recording_micResponseSongGen_2_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micResponseSongGen_2RecFolder, clipFilename))
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
