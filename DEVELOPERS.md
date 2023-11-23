# FARGOpy
## Wrapping FRAGO3D
### Developer notes

## Configuration and installing

The running and configuration of `FARGOpy` depends on the directory name `$HOME/.fargopy` 
and the files contained on it. This directory and its basic contain are created when 
you import for the first time the package in one of your files, or when you run `ifargopy` also 
for the first time.

This configuration directory contains the following files:

- `fargopyrc`: This is the configuration file of the package.

- `ifargopy.py`: This is the initialization script that is ran when you invoke `ifargopy`.

If you update `fargopy`, the next time you import it or call the script, it will ask you 
for changing the configuration file. It is a good idea to change it in case new variables
are added to the file.