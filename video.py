from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room
import random
import string

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Dictionary to store room IDs and their corresponding usernames
rooms = {}

@app.route('/')
def index():
    return render_template('video.html')

@app.route('/create_room')
def create_room():
    # Generate a random room ID
    room_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    # Add the room ID to the dictionary with an empty list of users
    rooms[room_id] = []
    return room_id

@socketio.on('join')
def join(message):
    username = message['username']
    room = message['room']
    join_room(room)
    rooms[room].append(username)  # Add the username to the list of users in the room
    emit('ready', {'username': username, 'users': rooms[room]}, to=room, skip_sid=request.sid)

@socketio.on('call')
def call(message):
    callee = message['callee']
    emit('call', {'caller': request.sid}, to=callee)

@socketio.on('accept')
def accept(message):
    caller = message['caller']
    emit('accept', {'callee': request.sid}, to=caller)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=8080)
