"""
Simple backend server to handle LiveKit room creation and agent dispatch
"""
import os
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS
from livekit import api
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# LiveKit credentials
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

@app.route('/create_room', methods=['POST'])
def create_room():
    """Create a room and dispatch an agent"""
    try:
        # Get user identity from request or generate one
        data = request.json or {}
        user_identity = data.get('identity', f'user-{os.urandom(4).hex()}')
        room_name = data.get('room', 'voice-assistant')
        
        # Run async operations in sync context
        result = asyncio.run(_create_room_async(room_name, user_identity))
        return jsonify(result)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

async def _create_room_async(room_name, user_identity):
    """Async function to create room and dispatch agent"""
    lkapi = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    
    try:
        # Create or get room
        try:
            room = await lkapi.room.create_room(api.CreateRoomRequest(name=room_name))
            print(f"✓ Room created: {room_name}")
        except Exception as e:
            print(f"Room might already exist: {e}")
        
        # Dispatch agent to the room
        try:
            await lkapi.agent.dispatch_room_agent(
                api.RoomAgentDispatch(
                    room=room_name,
                )
            )
            print(f"✓ Agent dispatched to room: {room_name}")
        except Exception as e:
            print(f"Agent dispatch info: {e}")
        
        # Generate token for user
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
            .with_identity(user_identity) \
            .with_name(f"User {user_identity}") \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            ))
        
        jwt_token = token.to_jwt()
        
        return {
            'success': True,
            'room': room_name,
            'token': jwt_token,
            'url': LIVEKIT_URL,
            'identity': user_identity
        }
    finally:
        await lkapi.aclose()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("=" * 60)
    print("LiveKit Backend Server")
    print("=" * 60)
    print(f"Server running on: http://localhost:5000")
    print(f"LiveKit URL: {LIVEKIT_URL}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)

