import pyglet				# Imports the module
pyglet.options['search_local_libs'] = True	# Allows pyglet to access files in your directory

my_music = pyglet.media.load("count.mp3")	# Loads the audio file you want to play (it does not have to be a .ogg

my_player = pyglet.media.Player()	# Creates a player

my_player.queue(my_music)	# Adds your song to the players queue
my_player.loop = False 		# If you want the player to loop your song. By default this value is False
my_player.play()			# Starts playing your audio file when you run the app

pyglet.app.run()			# Runs the app