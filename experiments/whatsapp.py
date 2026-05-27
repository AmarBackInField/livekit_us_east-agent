"""
WhatsApp Business API Integration

This module handles:
1. WhatsApp Business OAuth authentication
2. Getting User Access Token
3. Sending WhatsApp messages (text, templates, media)
4. Receiving messages via webhook
5. Managing phone numbers and contacts
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
from cryptography.fernet import Fernet

# Initialize FastAPI app
app = FastAPI(
    title="WhatsApp Business API Service",
    description="FastAPI service for WhatsApp Business OAuth and messaging",
    version="1.0.0"
)

# ============================================
# CONFIGURATION - Update these with your values
# ============================================

# Meta App Credentials (Same as Messenger - from https://developers.facebook.com/apps)
META_APP_ID = os.getenv('META_APP_ID', '860152246958629')
META_APP_SECRET = os.getenv('META_APP_SECRET', 'e846aabbbe1f948face1b1679fff9e19')

# Callback URL (must match what's configured in Meta Developer Console)
REDIRECT_URI = os.getenv('META_REDIRECT_URI', 'https://unnumbered-debasedly-cyndi.ngrok-free.dev/whatsapp/callback')

# MongoDB Configuration
MONGODB_URI = os.getenv(
    'MONGODB_URI',
    "mongodb+srv://pythonProd:pythonfindiy25@findiy-main.t5gfeq.mongodb.net/Findiy_Production_Python?retryWrites=true&w=majority&appName=Findiy-main"
)
DB_NAME = "Findiy_Production_Python"
COLLECTION_NAME = "whatsapp_tokens"
MESSAGES_COLLECTION = "whatsapp_messages"

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
messages_collection = db[MESSAGES_COLLECTION]

# Encryption key for storing tokens securely
DEFAULT_ENCRYPTION_KEY = "nybmG4fqyl5PZkymPJHsgBCCqxvf1jqwpENm-0-crVo="
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', DEFAULT_ENCRYPTION_KEY).encode()
cipher = Fernet(ENCRYPTION_KEY)

# Meta Graph API Base URL
GRAPH_API_URL = "https://graph.facebook.com/v18.0"

# WhatsApp Business API Scopes
SCOPES = [
    "whatsapp_business_management",
    "whatsapp_business_messaging",
    "business_management"
]

# Webhook Verify Token
VERIFY_TOKEN = os.getenv('WHATSAPP_VERIFY_TOKEN', 'whatsapp_verify_token_123')


# ============================================
# Pydantic Models
# ============================================

class SendTextMessageRequest(BaseModel):
    phone_number: str  # Recipient phone number with country code (e.g., "919876543210")
    message: str
    phone_number_id: str  # Your WhatsApp Business phone number ID


class SendTemplateMessageRequest(BaseModel):
    phone_number: str  # Recipient phone number with country code
    template_name: str  # Template name (e.g., "hello_world")
    language_code: str = "en_US"  # Language code
    phone_number_id: str  # Your WhatsApp Business phone number ID
    template_params: Optional[List[str]] = None  # Template parameters


class SendMediaMessageRequest(BaseModel):
    phone_number: str
    media_type: str  # "image", "video", "audio", "document"
    media_url: str  # URL of the media
    caption: Optional[str] = None
    phone_number_id: str


class MessageResponse(BaseModel):
    success: bool
    message_id: Optional[str] = None
    phone_number: str
    error: Optional[str] = None


class WhatsAppAccount(BaseModel):
    waba_id: str  # WhatsApp Business Account ID
    phone_number_id: str
    display_phone_number: str
    verified_name: str


class UserConnectionStatus(BaseModel):
    connected: bool
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    whatsapp_accounts: Optional[List[WhatsAppAccount]] = None


# ============================================
# Helper Functions
# ============================================

def encrypt_token(token: str) -> str:
    """Encrypt sensitive token data"""
    return cipher.encrypt(token.encode()).decode()


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt token data"""
    return cipher.decrypt(encrypted_token.encode()).decode()


def save_user_tokens(user_id: str, user_data: dict):
    """Save user and WhatsApp tokens to MongoDB"""
    encrypted_data = {
        'user_id': user_id,
        'user_name': user_data.get('user_name'),
        'access_token': encrypt_token(user_data['access_token']),
        'token_expires_at': user_data.get('token_expires_at'),
        'whatsapp_accounts': user_data.get('whatsapp_accounts', []),
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }
    
    collection.update_one(
        {'user_id': user_id},
        {'$set': encrypted_data},
        upsert=True
    )
    print(f"✅ WhatsApp tokens saved for user {user_id}")


def get_user_tokens(user_id: str) -> Optional[dict]:
    """Retrieve user tokens from MongoDB"""
    doc = collection.find_one({'user_id': user_id})
    
    if not doc:
        return None
    
    try:
        return {
            'user_id': doc['user_id'],
            'user_name': doc.get('user_name'),
            'access_token': decrypt_token(doc['access_token']),
            'token_expires_at': doc.get('token_expires_at'),
            'whatsapp_accounts': doc.get('whatsapp_accounts', [])
        }
    except Exception as e:
        print(f"Error decrypting tokens: {e}")
        return None


def get_access_token_by_phone_id(phone_number_id: str) -> Optional[str]:
    """Get access token for a specific phone number ID"""
    doc = collection.find_one({'whatsapp_accounts.phone_number_id': phone_number_id})
    
    if not doc:
        return None
    
    try:
        return decrypt_token(doc['access_token'])
    except:
        return None


def exchange_code_for_token(code: str) -> dict:
    """Exchange authorization code for access token"""
    token_url = f"{GRAPH_API_URL}/oauth/access_token"
    
    params = {
        'client_id': META_APP_ID,
        'client_secret': META_APP_SECRET,
        'redirect_uri': REDIRECT_URI,
        'code': code
    }
    
    response = requests.get(token_url, params=params)
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to exchange code for token: {response.json()}"
        )
    
    return response.json()


def get_long_lived_token(short_lived_token: str) -> dict:
    """Exchange short-lived token for long-lived token (60 days)"""
    token_url = f"{GRAPH_API_URL}/oauth/access_token"
    
    params = {
        'grant_type': 'fb_exchange_token',
        'client_id': META_APP_ID,
        'client_secret': META_APP_SECRET,
        'fb_exchange_token': short_lived_token
    }
    
    response = requests.get(token_url, params=params)
    
    if response.status_code != 200:
        return {'access_token': short_lived_token, 'expires_in': 3600}
    
    return response.json()


def get_user_info(access_token: str) -> dict:
    """Get user profile information"""
    url = f"{GRAPH_API_URL}/me"
    
    params = {
        'fields': 'id,name',
        'access_token': access_token
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to get user info: {response.json()}"
        )
    
    return response.json()


def get_whatsapp_business_accounts(access_token: str) -> List[dict]:
    """Get list of WhatsApp Business Accounts the user has access to"""
    # First, get the user's businesses
    businesses_url = f"{GRAPH_API_URL}/me/businesses"
    
    params = {
        'fields': 'id,name,owned_whatsapp_business_accounts',
        'access_token': access_token
    }
    
    response = requests.get(businesses_url, params=params)
    
    if response.status_code != 200:
        print(f"Failed to get businesses: {response.json()}")
        return []
    
    businesses = response.json().get('data', [])
    whatsapp_accounts = []
    
    for business in businesses:
        wabas = business.get('owned_whatsapp_business_accounts', {}).get('data', [])
        
        for waba in wabas:
            waba_id = waba['id']
            
            # Get phone numbers for this WABA
            phone_numbers_url = f"{GRAPH_API_URL}/{waba_id}/phone_numbers"
            phone_params = {
                'fields': 'id,display_phone_number,verified_name,quality_rating',
                'access_token': access_token
            }
            
            phone_response = requests.get(phone_numbers_url, params=phone_params)
            
            if phone_response.status_code == 200:
                phones = phone_response.json().get('data', [])
                
                for phone in phones:
                    whatsapp_accounts.append({
                        'waba_id': waba_id,
                        'phone_number_id': phone['id'],
                        'display_phone_number': phone.get('display_phone_number', ''),
                        'verified_name': phone.get('verified_name', ''),
                        'quality_rating': phone.get('quality_rating', '')
                    })
    
    return whatsapp_accounts


def save_incoming_message(message_data: dict):
    """Save incoming WhatsApp message to database"""
    messages_collection.insert_one({
        **message_data,
        'received_at': datetime.utcnow()
    })


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "WhatsApp Business API Service",
        "version": "1.0.0",
        "endpoints": {
            "authorize": "/whatsapp/login - Start OAuth flow",
            "callback": "/whatsapp/callback - OAuth callback (automatic)",
            "status": "/whatsapp/status/{user_id} - Check connection status",
            "accounts": "/api/whatsapp-accounts/{user_id} - List WhatsApp accounts",
            "send_text": "/api/send-text - Send text message",
            "send_template": "/api/send-template - Send template message",
            "send_media": "/api/send-media - Send media message",
            "contacts": "/api/contacts/{phone_number_id} - List contacts",
            "webhook": "/webhook - WhatsApp webhook endpoint"
        },
        "setup_instructions": {
            "step_1": "Configure META_APP_ID and META_APP_SECRET",
            "step_2": "Add WhatsApp product to your Meta App",
            "step_3": "Set REDIRECT_URI to match Meta App callback URL",
            "step_4": "Visit /whatsapp/login to start the OAuth flow",
            "step_5": "Use /api/send-text or /api/send-template to send messages"
        }
    }


@app.get("/whatsapp/login")
async def authorize():
    """
    Step 1: Redirect user to Facebook Login Dialog for WhatsApp permissions
    """
    auth_url = "https://www.facebook.com/v18.0/dialog/oauth"
    
    scope_string = ",".join(SCOPES)
    
    params = {
        'client_id': META_APP_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': scope_string,
        'response_type': 'code',
        'state': 'whatsapp_auth_state'
    }
    
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    full_auth_url = f"{auth_url}?{query_string}"
    
    return RedirectResponse(url=full_auth_url)


@app.get("/whatsapp/callback")
async def oauth_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None
):
    """
    Step 2: Handle OAuth callback from Facebook
    """
    if error:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": error,
                "error_description": error_description
            }
        )
    
    if not code:
        raise HTTPException(status_code=400, detail="No authorization code received")
    
    try:
        # Exchange code for token
        token_data = exchange_code_for_token(code)
        short_lived_token = token_data['access_token']
        
        # Get long-lived token
        long_lived_data = get_long_lived_token(short_lived_token)
        access_token = long_lived_data['access_token']
        expires_in = long_lived_data.get('expires_in', 5184000)
        
        # Get user info
        user_info = get_user_info(access_token)
        user_id = user_info['id']
        user_name = user_info.get('name', 'Unknown')
        
        # Get WhatsApp Business Accounts
        whatsapp_accounts = get_whatsapp_business_accounts(access_token)
        
        # Save to database
        user_data = {
            'user_name': user_name,
            'access_token': access_token,
            'token_expires_at': datetime.utcnow() + timedelta(seconds=expires_in),
            'whatsapp_accounts': whatsapp_accounts
        }
        
        save_user_tokens(user_id, user_data)
        
        return {
            "success": True,
            "message": "WhatsApp Business account connected successfully!",
            "user_id": user_id,
            "user_name": user_name,
            "whatsapp_accounts_connected": len(whatsapp_accounts),
            "accounts": [
                {
                    "phone_number_id": acc['phone_number_id'],
                    "display_phone_number": acc['display_phone_number'],
                    "verified_name": acc['verified_name']
                }
                for acc in whatsapp_accounts
            ],
            "note": "Use phone_number_id in API requests to send messages."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OAuth callback error: {str(e)}")


@app.get("/whatsapp/status/{user_id}", response_model=UserConnectionStatus)
async def check_connection_status(user_id: str):
    """Check if a user is connected and get their WhatsApp accounts"""
    user_data = get_user_tokens(user_id)
    
    if not user_data:
        return UserConnectionStatus(connected=False)
    
    accounts = [
        WhatsAppAccount(
            waba_id=acc['waba_id'],
            phone_number_id=acc['phone_number_id'],
            display_phone_number=acc['display_phone_number'],
            verified_name=acc['verified_name']
        )
        for acc in user_data.get('whatsapp_accounts', [])
    ]
    
    return UserConnectionStatus(
        connected=True,
        user_id=user_data['user_id'],
        user_name=user_data.get('user_name'),
        whatsapp_accounts=accounts
    )


@app.get("/api/whatsapp-accounts/{user_id}")
async def list_whatsapp_accounts(user_id: str):
    """List all WhatsApp Business accounts for a user"""
    user_data = get_user_tokens(user_id)
    
    if not user_data:
        raise HTTPException(
            status_code=404,
            detail="User not found. Please authorize at /whatsapp/login"
        )
    
    return {
        "user_id": user_id,
        "user_name": user_data.get('user_name'),
        "total_accounts": len(user_data.get('whatsapp_accounts', [])),
        "accounts": user_data.get('whatsapp_accounts', [])
    }


@app.post("/api/send-text", response_model=MessageResponse)
async def send_text_message(request: SendTextMessageRequest):
    """
    Send a text message via WhatsApp
    
    - **phone_number**: Recipient's phone number with country code (e.g., "919876543210")
    - **message**: The text message to send
    - **phone_number_id**: Your WhatsApp Business phone number ID
    """
    access_token = get_access_token_by_phone_id(request.phone_number_id)
    
    if not access_token:
        raise HTTPException(
            status_code=404,
            detail=f"Phone number ID {request.phone_number_id} not found. Please authorize at /whatsapp/login"
        )
    
    # WhatsApp Cloud API endpoint
    send_url = f"{GRAPH_API_URL}/{request.phone_number_id}/messages"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'messaging_product': 'whatsapp',
        'recipient_type': 'individual',
        'to': request.phone_number,
        'type': 'text',
        'text': {
            'preview_url': False,
            'body': request.message
        }
    }
    
    try:
        response = requests.post(send_url, headers=headers, json=payload)
        result = response.json()
        
        if response.status_code == 200:
            message_id = result.get('messages', [{}])[0].get('id')
            return MessageResponse(
                success=True,
                message_id=message_id,
                phone_number=request.phone_number
            )
        else:
            error_msg = result.get('error', {}).get('message', 'Unknown error')
            return MessageResponse(
                success=False,
                phone_number=request.phone_number,
                error=error_msg
            )
            
    except Exception as e:
        return MessageResponse(
            success=False,
            phone_number=request.phone_number,
            error=str(e)
        )


@app.post("/api/send-template", response_model=MessageResponse)
async def send_template_message(request: SendTemplateMessageRequest):
    """
    Send a template message via WhatsApp
    
    Template messages can be sent outside the 24-hour window.
    Templates must be pre-approved by WhatsApp.
    
    - **phone_number**: Recipient's phone number with country code
    - **template_name**: The approved template name
    - **language_code**: Language code (default: "en_US")
    - **phone_number_id**: Your WhatsApp Business phone number ID
    - **template_params**: Optional list of template parameters
    """
    access_token = get_access_token_by_phone_id(request.phone_number_id)
    
    if not access_token:
        raise HTTPException(
            status_code=404,
            detail=f"Phone number ID {request.phone_number_id} not found"
        )
    
    send_url = f"{GRAPH_API_URL}/{request.phone_number_id}/messages"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    template_data = {
        'name': request.template_name,
        'language': {
            'code': request.language_code
        }
    }
    
    # Add template parameters if provided
    if request.template_params:
        template_data['components'] = [
            {
                'type': 'body',
                'parameters': [
                    {'type': 'text', 'text': param}
                    for param in request.template_params
                ]
            }
        ]
    
    payload = {
        'messaging_product': 'whatsapp',
        'recipient_type': 'individual',
        'to': request.phone_number,
        'type': 'template',
        'template': template_data
    }
    
    try:
        response = requests.post(send_url, headers=headers, json=payload)
        result = response.json()
        
        if response.status_code == 200:
            message_id = result.get('messages', [{}])[0].get('id')
            return MessageResponse(
                success=True,
                message_id=message_id,
                phone_number=request.phone_number
            )
        else:
            error_msg = result.get('error', {}).get('message', 'Unknown error')
            return MessageResponse(
                success=False,
                phone_number=request.phone_number,
                error=error_msg
            )
            
    except Exception as e:
        return MessageResponse(
            success=False,
            phone_number=request.phone_number,
            error=str(e)
        )


@app.post("/api/send-media", response_model=MessageResponse)
async def send_media_message(request: SendMediaMessageRequest):
    """
    Send a media message via WhatsApp
    
    - **phone_number**: Recipient's phone number with country code
    - **media_type**: Type of media ("image", "video", "audio", "document")
    - **media_url**: Public URL of the media file
    - **caption**: Optional caption for the media
    - **phone_number_id**: Your WhatsApp Business phone number ID
    """
    access_token = get_access_token_by_phone_id(request.phone_number_id)
    
    if not access_token:
        raise HTTPException(
            status_code=404,
            detail=f"Phone number ID {request.phone_number_id} not found"
        )
    
    send_url = f"{GRAPH_API_URL}/{request.phone_number_id}/messages"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    media_object = {'link': request.media_url}
    if request.caption and request.media_type in ['image', 'video', 'document']:
        media_object['caption'] = request.caption
    
    payload = {
        'messaging_product': 'whatsapp',
        'recipient_type': 'individual',
        'to': request.phone_number,
        'type': request.media_type,
        request.media_type: media_object
    }
    
    try:
        response = requests.post(send_url, headers=headers, json=payload)
        result = response.json()
        
        if response.status_code == 200:
            message_id = result.get('messages', [{}])[0].get('id')
            return MessageResponse(
                success=True,
                message_id=message_id,
                phone_number=request.phone_number
            )
        else:
            error_msg = result.get('error', {}).get('message', 'Unknown error')
            return MessageResponse(
                success=False,
                phone_number=request.phone_number,
                error=error_msg
            )
            
    except Exception as e:
        return MessageResponse(
            success=False,
            phone_number=request.phone_number,
            error=str(e)
        )


@app.get("/api/contacts/{phone_number_id}")
async def get_contacts(phone_number_id: str):
    """
    Get contacts/conversations for a WhatsApp Business number
    
    Note: WhatsApp doesn't have a direct "contacts" API like Messenger.
    This endpoint returns stored incoming messages to show who has contacted you.
    """
    # Get unique senders from stored messages
    pipeline = [
        {'$match': {'phone_number_id': phone_number_id}},
        {'$group': {
            '_id': '$from_number',
            'name': {'$first': '$from_name'},
            'last_message': {'$last': '$message'},
            'last_contact': {'$max': '$received_at'},
            'message_count': {'$sum': 1}
        }},
        {'$sort': {'last_contact': -1}}
    ]
    
    contacts = list(messages_collection.aggregate(pipeline))
    
    formatted_contacts = [
        {
            'phone_number': contact['_id'],
            'name': contact.get('name', 'Unknown'),
            'last_message': contact.get('last_message', ''),
            'last_contact': contact.get('last_contact'),
            'message_count': contact.get('message_count', 0)
        }
        for contact in contacts
    ]
    
    return {
        "phone_number_id": phone_number_id,
        "total_contacts": len(formatted_contacts),
        "contacts": formatted_contacts
    }


@app.get("/api/messages/{phone_number_id}")
async def get_messages(
    phone_number_id: str,
    contact_number: Optional[str] = None,
    limit: int = 50
):
    """
    Get message history for a WhatsApp Business number
    
    - **phone_number_id**: Your WhatsApp Business phone number ID
    - **contact_number**: Optional - filter by specific contact
    - **limit**: Maximum messages to return (default 50)
    """
    query = {'phone_number_id': phone_number_id}
    
    if contact_number:
        query['from_number'] = contact_number
    
    messages = list(
        messages_collection.find(query)
        .sort('received_at', -1)
        .limit(limit)
    )
    
    # Convert ObjectId to string for JSON serialization
    for msg in messages:
        msg['_id'] = str(msg['_id'])
    
    return {
        "phone_number_id": phone_number_id,
        "contact_filter": contact_number,
        "total_messages": len(messages),
        "messages": messages
    }


@app.get("/webhook")
async def webhook_verify(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    """
    Webhook verification endpoint for WhatsApp
    
    Configure in Meta Developer Console:
    - Callback URL: https://your-domain/webhook
    - Verify Token: whatsapp_verify_token_123
    """
    if hub_mode == 'subscribe' and hub_verify_token == VERIFY_TOKEN:
        print("✅ WhatsApp Webhook verified!")
        return int(hub_challenge)
    
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def webhook_receive(request: Request):
    """
    Webhook endpoint to receive messages from WhatsApp
    
    When someone sends a message to your WhatsApp Business number,
    it will be received here.
    """
    try:
        body = await request.json()
        print(f"📩 Received WhatsApp webhook: {body}")
        
        # Process incoming messages
        if body.get('object') == 'whatsapp_business_account':
            for entry in body.get('entry', []):
                for change in entry.get('changes', []):
                    value = change.get('value', {})
                    
                    # Get phone number ID
                    phone_number_id = value.get('metadata', {}).get('phone_number_id')
                    
                    # Process messages
                    for message in value.get('messages', []):
                        from_number = message.get('from')
                        message_id = message.get('id')
                        timestamp = message.get('timestamp')
                        message_type = message.get('type')
                        
                        # Get sender name from contacts
                        contacts = value.get('contacts', [])
                        from_name = contacts[0].get('profile', {}).get('name', 'Unknown') if contacts else 'Unknown'
                        
                        # Extract message content based on type
                        content = ''
                        if message_type == 'text':
                            content = message.get('text', {}).get('body', '')
                        elif message_type == 'image':
                            content = f"[Image] {message.get('image', {}).get('caption', '')}"
                        elif message_type == 'video':
                            content = f"[Video] {message.get('video', {}).get('caption', '')}"
                        elif message_type == 'audio':
                            content = "[Audio message]"
                        elif message_type == 'document':
                            content = f"[Document] {message.get('document', {}).get('filename', '')}"
                        elif message_type == 'location':
                            loc = message.get('location', {})
                            content = f"[Location] {loc.get('latitude')}, {loc.get('longitude')}"
                        
                        print(f"📱 Message from {from_name} ({from_number}): {content}")
                        
                        # Save message to database
                        save_incoming_message({
                            'phone_number_id': phone_number_id,
                            'message_id': message_id,
                            'from_number': from_number,
                            'from_name': from_name,
                            'message_type': message_type,
                            'message': content,
                            'timestamp': timestamp,
                            'raw_message': message
                        })
                    
                    # Process status updates
                    for status in value.get('statuses', []):
                        print(f"📊 Message status: {status.get('id')} - {status.get('status')}")
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/templates/{phone_number_id}")
async def get_message_templates(phone_number_id: str):
    """
    Get available message templates for a WhatsApp Business number
    """
    # Find the WABA ID for this phone number
    doc = collection.find_one({'whatsapp_accounts.phone_number_id': phone_number_id})
    
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Phone number ID {phone_number_id} not found"
        )
    
    # Get WABA ID and access token
    waba_id = None
    for acc in doc.get('whatsapp_accounts', []):
        if acc['phone_number_id'] == phone_number_id:
            waba_id = acc['waba_id']
            break
    
    if not waba_id:
        raise HTTPException(status_code=404, detail="WABA ID not found")
    
    try:
        access_token = decrypt_token(doc['access_token'])
    except:
        raise HTTPException(status_code=401, detail="Invalid access token")
    
    # Fetch templates from Graph API
    templates_url = f"{GRAPH_API_URL}/{waba_id}/message_templates"
    
    params = {
        'fields': 'name,status,category,language,components',
        'access_token': access_token
    }
    
    response = requests.get(templates_url, params=params)
    result = response.json()
    
    if response.status_code != 200:
        error_msg = result.get('error', {}).get('message', 'Unknown error')
        raise HTTPException(status_code=400, detail=f"Graph API error: {error_msg}")
    
    templates = result.get('data', [])
    
    return {
        "phone_number_id": phone_number_id,
        "waba_id": waba_id,
        "total_templates": len(templates),
        "templates": templates
    }


@app.get("/api/connected-users")
async def list_connected_users():
    """List all connected WhatsApp Business accounts (Admin endpoint)"""
    try:
        users = list(collection.find({}, {'_id': 0, 'access_token': 0}))
        
        return {
            "total_users": len(users),
            "users": users
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing users: {str(e)}")


class UpdateTokenRequest(BaseModel):
    access_token: str
    phone_number_id: str


@app.post("/api/update-token")
async def update_access_token(request: UpdateTokenRequest):
    """
    Manually update the access token for a WhatsApp phone number
    
    Use this to set the token generated from Meta Developer Console.
    
    - **access_token**: The access token from Meta Developer Console
    - **phone_number_id**: Your WhatsApp Business phone number ID
    """
    # Find the document with this phone number
    doc = collection.find_one({'whatsapp_accounts.phone_number_id': request.phone_number_id})
    
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Phone number ID {request.phone_number_id} not found"
        )
    
    # Update the access token
    encrypted_token = encrypt_token(request.access_token)
    
    collection.update_one(
        {'whatsapp_accounts.phone_number_id': request.phone_number_id},
        {
            '$set': {
                'access_token': encrypted_token,
                'updated_at': datetime.utcnow()
            }
        }
    )
    
    return {
        "success": True,
        "message": "Access token updated successfully",
        "phone_number_id": request.phone_number_id
    }


@app.delete("/api/disconnect/{user_id}")
async def disconnect_user(user_id: str):
    """Disconnect a user and remove their stored tokens"""
    result = collection.delete_one({'user_id': user_id})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found"
        )
    
    return {
        "success": True,
        "message": f"User {user_id} disconnected successfully"
    }


# ============================================
# Run the application
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🚀 Starting WhatsApp Business API Service")
    print("=" * 60)
    print(f"📱 App ID: {META_APP_ID}")
    print(f"🔗 Callback URL: {REDIRECT_URI}")
    print(f"🔐 Webhook Verify Token: {VERIFY_TOKEN}")
    print("=" * 60)
    print("📝 API Documentation: http://localhost:8000/docs")
    print("🔄 ReDoc: http://localhost:8000/redoc")
    print("=" * 60)
    print("\n⚠️  Setup Requirements:")
    print("   1. Add 'WhatsApp' product to your Meta App")
    print("   2. Configure webhook URL in Meta Developer Console")
    print("   3. Link a WhatsApp Business Account")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)