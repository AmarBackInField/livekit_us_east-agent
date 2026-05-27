"""
LiveKit Token Generator
Generates an access token for connecting to your LiveKit room
"""
import os
from livekit import api
from dotenv import load_dotenv

load_dotenv()

def generate_token():
    """Generate a LiveKit access token"""
    
    # Your LiveKit credentials from .env
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    livekit_url = os.getenv("LIVEKIT_URL")
    
    if not api_key or not api_secret:
        print("❌ Error: LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in .env file")
        return
    
    # Create token with permissions
    token = api.AccessToken(api_key, api_secret) \
        .with_identity("web-user") \
        .with_name("Web User") \
        .with_grants(api.VideoGrants(
            room_join=True,
            room="voice-assistant",
            can_publish=True,
            can_subscribe=True,
        ))
    
    jwt_token = token.to_jwt()
    
    print("\n" + "="*80)
    print("LIVEKIT ACCESS TOKEN GENERATED")
    print("="*80)
    print(f"\nWebSocket URL: {livekit_url}")
    print(f"\nAccess Token:\n{jwt_token}")
    print("\n" + "="*80)
    print("\nCopy the token above and paste it into the frontend's 'Access Token' field")
    print("This token is valid for 24 hours\n")
    
    return jwt_token

if __name__ == "__main__":
    generate_token()

