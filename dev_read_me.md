
# To create a proposed change.

## Create a new branch
Name it something that describes the improvements/fixes you want to make. e.g "home-screen-styling" or "fraud-detection-content"

The command is `git checkout -b branch-name`

## Development and Visualization

Make your changes by editing the HTML files. Note that `index.html` is the home page for the app. Other HTMLs throughout this directory are served up and linked from `index.html` as well. 

Visualizing changes is easy: just write `open index.html` to view the current changes in your favorite browser.

#### To test on mobile. 
Old Way: Push the change then wait for it to deploy and then test on mobile.
New Way: 

In the Terminal, on MacOS, run `ipconfig getifaddr en0` to get the IP Address of your local computer. 

You'll get an IP address `(host_ip_address)` resembling `10.0.0.229`

Now in the root directory of this website run `python -m http.server 8001` to serve the files up. 

While both the mobile device and your computer are connected to the same WiFi:
in a browser on your mobile device, visit `http://{host_ip_address}:8001/index.html`

The website should load. 


## To commit changes

add, commit and push changes to the new branch. 
Then in Github UI on the browser create a Pull Request merging your `branch-name` into main. 
Request me (Sam) as a reviewer and I will approve or request changes and then eventually approve. 
Once I approve, you will merge into main and that will be the new working version!
