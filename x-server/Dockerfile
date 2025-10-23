# Pull the baseimage.
FROM jlesage/baseimage-gui:alpine-3.22-v4

# Install xterm.
RUN add-pkg xterm

# Add Python 3.12 and dependencies
# The 'python3' package on Alpine is currently 3.12.x
# The 'py3-pip' package is needed for the pip package manager
RUN add-pkg python3 py3-pip

# Copy the start script.
COPY startapp.sh /startapp.sh

# Set the application name.
RUN set-cont-env APP_NAME "Xterm"