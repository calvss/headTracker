# headTracker
A computer vision based head tracker using virtual joystick.  


This program uses your computer webcam to track the angle of your head by detecting two colored squares attached to your headset.  

Attach one square to a spot near the ear, and another square to a mic arm. The program will then use a threshold of the hues of the camera to determine which spots are the colored squares. The program will calculate the parallax change to determine the angle of the two squares. 

POV Hat #1 on virtual joystick device 1 will be controlled by the program.  

You can type commands in the console window to change the program settings.  
Statistics from `s` can show the current hue under the mouse cursor when the mouse is on the 'hues' window.  


## Command syntax: [acfhloqsuvzpid] ['value']  
	h          : help
	q          : quit
	s          : toggle statistics
	u [int]    : upper threshold (default 170)
	l [int]    : lower threshold (default 130)
	a [double] : low-pass filter constant (default 0.3)
	z [int]    : deadzone in centidegrees (default 100, might not work as of 1.0)
	f [int]    : vJoy controller update rate (default 60)
	v [int]    : sensitivity (default 0.2)
	o [int]    : offset (default -500)
	c [int]    : computer vision framerate (default 24, heavily affected by CPU)
	p [double] : PID controller kP (default 0.9)
	i [double] : PID controller kI (default 0.1)
	d [double] : PID controller kD (default 2.0)


Runtime dependencies: virtual joystick 2.1.9 or newer

Build dependencies:
* virtual joystick SDK 2.1.9
* openCV 4.5.0

Copyright 2020 by Calvin Ng  
Released under the MIT License    
