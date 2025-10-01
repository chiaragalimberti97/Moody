Python 3.11.4 o 3.11.2

In order to use the web app you need to create a virtual environment in the Desktop: 
	python -m venv app.venv


Then you need to activate it, enter it : 

	app.venv\Scripts\activate 
	cd '.\app.venv\Web App - Moody\'

Then Install the necessary requirements: 

	pip install -r requirements.txt

Add your Spotify account by:
	
	- go to Spotify Developer Dashboard: https://developer.spotify.com/dashboard
	- create a new app ( give a name and description)
	- Save the client id e client secret
	- as redirect uri write: http://127.0.0.1:5000/callback
	- open personal_data.env and insert the client_id and client_secret in the apposite sections

And finally call the function main.py: 

	flask --app main.py run


The results :
![Enrollment](README_images\image1.png)
![Website structure](README_images\image2.png)
![Spotify player](README_images\image3.png)

