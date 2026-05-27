from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MongoDBService:
    """
    Service for managing voice agent settings in MongoDB.
    """
    
    def __init__(
        self,
        uri: str = "mongodb+srv://LOVJEET:LOVJEETMONGO@cluster0.zpzj90m.mongodb.net/?retryWrites=true&w=majority",
        database_name: str = "inshora_website",
        collection_name: str = "voice_agent"
    ):
        """
        Initialize MongoDB connection.
        
        Args:
            uri: MongoDB connection URI
            database_name: Name of the database
            collection_name: Name of the collection
        """
        self.uri = uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        
        try:
            self.connect()
            logger.info(f"✓ Connected to MongoDB: {database_name}.{collection_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    
    def load_settings(self) -> Optional[Dict[str, Any]]:
        """
        Load voice agent settings from MongoDB.
        Returns the first (and only) document in the collection.
        
        Returns:
            Dictionary containing voice agent settings, or None if not found
        """
        try:
            document = self.collection.find_one()
            
            if document:
                # Remove MongoDB's _id field for cleaner output
                if '_id' in document:
                    document['_id'] = str(document['_id'])
                
                logger.info("✓ Voice agent settings loaded from MongoDB")
                return document
            else:
                logger.warning("No voice agent settings found in MongoDB")
                return None
                
        except OperationFailure as e:
            logger.error(f"Failed to load settings: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading settings: {e}")
            return None
    
    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Update voice agent settings in MongoDB.
        If no document exists, creates one. Otherwise updates the existing document.
        
        Args:
            settings: Dictionary containing voice agent settings to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove _id if present in settings to avoid update conflicts
            settings_copy = settings.copy()
            if '_id' in settings_copy:
                del settings_copy['_id']
            
            # Check if document exists
            existing_doc = self.collection.find_one()
            
            if existing_doc:
                # Update existing document
                result = self.collection.update_one(
                    {"_id": existing_doc["_id"]},
                    {"$set": settings_copy}
                )
                logger.info(f"✓ Voice agent settings updated (matched: {result.matched_count}, modified: {result.modified_count})")
                return result.matched_count > 0
            else:
                # Insert new document
                result = self.collection.insert_one(settings_copy)
                logger.info(f"✓ Voice agent settings created with ID: {result.inserted_id}")
                return result.inserted_id is not None
                
        except OperationFailure as e:
            logger.error(f"Failed to update settings: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating settings: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a specific setting value by key.
        
        Args:
            key: Setting key to retrieve
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        settings = self.load_settings()
        if settings:
            return settings.get(key, default)
        return default
    
    def delete_settings(self) -> bool:
        """
        Delete all voice agent settings (clear the collection).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.collection.delete_many({})
            logger.info(f"✓ Deleted {result.deleted_count} voice agent settings documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete settings: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Default settings structure for reference
DEFAULT_VOICE_SETTINGS = {
    "voice_id": "694f9389-aac1-45b6-b726-9d9369183238",  # Default Cartesia voice ID
    "prompt": """You are a voice AI assistant named Kelly from Inshora Group. 
Be concise, friendly, and conversational.

Inshora Group is an insurance company based in Texas, USA.

You have access to a knowledge base containing all relevant and verified information about Inshora Group.
The most relevant context from the knowledge base will be provided to you in the chat history before each of your turns.
You MUST use this provided context to answer any user query related to Inshora Group, its services, policies, or operations.

If the provided context is insufficient or the user's query is generic, use your general knowledge.""",
    "llm_model": "gemini-2.0-flash-exp",
    "llm_temperature": 0.3,
    "stt_model": "nova-3",
    "stt_language": "en",
    "tts_model": "sonic-3",
    "tts_language": "en",
    "tts_sample_rate": 16000,
    "vad_min_speech_duration": 0.05,
    "vad_min_silence_duration": 0.2,
    "rag_collection": "inshora",
    "rag_top_k": 3,
    "rag_timeout": 0.75
}

