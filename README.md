# DumbMemes
A project that uses webcam to detect facial expression and display dumb memes. 
Inspiration - /reinesana/MeowCV

## What i Plan 
I plan to create a fun project to enjoy and learn coding. This project is aimed to display meme's we often find ourself using in chat, and humour our expression. 

I plan to display as meme's, which would require me to find the unique facial expression to trigger a specific meme. 

NOTE : This project uses mediapipe.tasks library for facemesh instead of .solutions 

# HOW IT WORKS

i. The project will use your webcam in order to retrieve face and using mediapipe .tasks library will track your expressions.

ii. An expression satifying certain threshold will trigger a MEME as a corresponding responce, for instance: 

   1. SMILE -> SMILING CAT
   2. WIDE OPEN EYES + MOUTH COVERED -> SHOCKED SPEED
   3. EYES CLOSE + SMILE -> SMILING SPEED
   etc...

# FUTURE WORKS

Some of the expression would require a complex approach to how the expression is to be captured.

An expression that that has a similar facial approach to another might not be reflective, as the pipe would priortize in displaying the other image. Eg - Shocked_Dog image is foreshowed by the Shocked_speed image.















