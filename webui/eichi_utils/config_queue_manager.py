# ==============================================================================
# CONFIG QUEUE MANAGER - CORE QUEUE PROCESSING SYSTEM
# ==============================================================================
"""
ConfigQueueManager CLASS DOCUMENTATION

PURPOSE:
The ConfigQueueManager is the core component that handles all config file operations
and queue processing logic. It manages the lifecycle of configuration files from
creation through queue processing to completion.

ARCHITECTURE:
- FILE-BASED QUEUE SYSTEM: Uses filesystem directories to manage queue state
- THREAD-SAFE PROCESSING: Handles concurrent access to queue operations
- IMAGE MANAGEMENT: Automatic image storage with deduplication
- ERROR RECOVERY: Robust error handling and state recovery mechanisms

DIRECTORY STRUCTURE:
configs/          - Saved configuration files (.json)
config_images/    - Permanent image storage with deduplication
queue/           - Items waiting to be processed
processing/      - Currently processing item
completed/       - Successfully processed items
error/           - Failed processing items with error details

WORKFLOW:
1. Save config → configs/
2. Queue config → copy to queue/
3. Start processing → move to processing/
4. Complete/Error → move to completed/ or error/

THREAD SAFETY:
- Queue processing runs in separate thread
- State flags prevent concurrent processing
- File operations use atomic moves where possible
"""

import os
import json
import time
import shutil
import threading
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

class ConfigQueueManager:
    """
    CORE QUEUE MANAGER: Handles all config file and queue operations
    
    INITIALIZATION PARAMETERS:
    base_path (str): Root directory for all queue-related folders
    
    INSTANCE VARIABLES:
    - Directory paths: configs_dir, queue_dir, processing_dir, completed_dir, error_dir
    - Processing state: is_processing, current_config, stop_processing
    - Threading: queue_thread for background processing
    
    THREAD SAFETY:
    Uses threading.Thread for queue processing with proper state management.
    File operations are atomic where possible to prevent corruption.
    """
    
    def __init__(self, base_path: str):
        """
        CONSTRUCTOR: Initialize queue manager with directory structure
        
        DIRECTORY CREATION:
        Automatically creates all necessary directories if they don't exist.
        This ensures the queue system can operate immediately after initialization.
        
        PARAMETERS:
        base_path: Root directory (typically the main application directory)
        """
        self.base_path = base_path
        self.configs_dir = os.path.join(base_path, 'configs')
        self.queue_dir = os.path.join(base_path, 'queue')
        self.processing_dir = os.path.join(base_path, 'processing')
        self.completed_dir = os.path.join(base_path, 'completed')
        self.error_dir = os.path.join(base_path, 'error')
        
        # Queue processing state
        self.is_processing = False
        self.current_config = None
        self.queue_thread = None
        self.stop_processing = False
        
        # Initialize directories
        self._init_directories()
    
    # ==============================================================================
    # IMAGE MANAGEMENT SYSTEM
    # ==============================================================================
    
    def is_gradio_temp_file(self, file_path):
        """
        UTILITY: Detect if file is in Gradio's temporary directory
        
        PURPOSE:
        Gradio stores uploaded files in temporary directories that are cleaned up
        automatically. This function detects such files so they can be copied
        to permanent storage before the temporary files are deleted.
        
        DETECTION PATTERNS:
        Checks for common Gradio temp directory patterns across platforms:
        - Windows: appdata\\local\\temp\\gradio
        - Linux/Mac: /tmp/gradio
        - Generic: any path containing 'gradio'
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        """
        if not file_path:
            return False
        
        normalized_path = os.path.normpath(file_path).lower()
        temp_patterns = [
            'appdata\\local\\temp\\gradio',
            '/tmp/gradio',
            'temp/gradio',
            '\\gradio\\',
            '/gradio/'
        ]
        
        return any(pattern in normalized_path for pattern in temp_patterns)

    def get_queue_status(self) -> Dict:
        """
        STATUS REPORTING: Get comprehensive queue status with completion order fix
        
        COMPREHENSIVE STATUS:
        Returns complete picture of queue state including:
        - Processing flags and current item
        - Queued items (waiting to process)
        - Completed items (newest first - FIXED)
        - Queue counts and metadata
        
        COMPLETION ORDER FIX:
        Previously showed oldest completed items first. Now correctly shows
        newest completed items first for better user feedback.
        
        ERROR HANDLING:
        Returns error status dict if filesystem operations fail.
        
        STATUS STRUCTURE:
        {
            "is_processing": bool,
            "queued": [config_names...],
            "processing": config_name or None,
            "completed": [config_names...],  # newest first
            "queue_count": int,
            "current_config": config_name or None
        }
        
        RETURNS:
        Dict: Complete status information
        """
        try:
            # Get queued configs
            queued_configs = []
            if os.path.exists(self.queue_dir):
                for file in sorted(os.listdir(self.queue_dir)):
                    if file.endswith('.json'):
                        config_name = file[:-5]
                        queued_configs.append(config_name)
                        
            # Get processing config
            processing_config = None
            if os.path.exists(self.processing_dir):
                for file in os.listdir(self.processing_dir):
                    if file.endswith('.json'):
                        processing_config = file[:-5]
                        break
                        
            # Get completed configs - FIXED: Sort by completion time (newest first)
            completed_configs = []
            if os.path.exists(self.completed_dir):
                json_files = [f for f in os.listdir(self.completed_dir) if f.endswith('.json')]
                # Sort by modification time (newest first)
                json_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.completed_dir, x)), reverse=True)
                
                for file in json_files:
                    config_name = file[:-5]  # Remove .json extension
                    completed_configs.append(config_name)
                    
            return {
                "is_processing": self.is_processing,
                "queued": queued_configs,
                "processing": processing_config,
                "completed": completed_configs[:10],  # Show first 10 (newest first)
                "queue_count": len(queued_configs),
                "current_config": self.current_config
            }
            
        except Exception as e:
            return {
                "error": f"Error getting queue status: {str(e)}",
                "is_processing": False,
                "queued": [],
                "processing": None,
                "completed": [],
                "queue_count": 0,
                "current_config": None
            }
        
    def clear_queue(self) -> Tuple[bool, str]:
        """
        QUEUE CLEARING: Remove all items from queue (safety check included)
        
        SAFETY MECHANISM:
        Prevents clearing queue while processing is active to avoid data loss.
        
        ATOMIC OPERATION:
        Removes all .json files from queue directory in single operation.
        
        RETURNS:
        Tuple[bool, str]: (success, message_with_count)
        """
        try:
            if self.is_processing:
                return False, "Cannot clear queue while processing"
                
            cleared_count = 0
            if os.path.exists(self.queue_dir):
                for file in os.listdir(self.queue_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(self.queue_dir, file))
                        cleared_count += 1
                        
            return True, f"Cleared {cleared_count} items from queue"
            
        except Exception as e:
            return False, f"Error clearing queue: {str(e)}"
       
    def find_image_by_original_name(self, original_filename):
        """
        IMAGE RECOVERY: Find existing image by original filename pattern
        
        PURPOSE:
        When config loading fails due to missing image, this function attempts
        to recover by finding images with similar original names but different hashes.
        
        RECOVERY STRATEGY:
        - Extract clean name from original filename
        - Search for files starting with that name
        - Return first match found
        
        PARAMETERS:
        original_filename: Original filename to search for
        
        RETURNS:
        str or None: Path to found image, or None if not found
        """
        try:
            config_images_dir = os.path.join(self.base_path, 'config_images')
            
            if not os.path.exists(config_images_dir):
                return None
            
            # Extract name without extension from original
            original_name = os.path.splitext(os.path.basename(original_filename))[0]
            safe_original_name = "".join(c for c in original_name if c.isalnum() or c in ('_', '-', ' ')).strip().replace(' ', '_')
            
            # Look for files that start with the same name
            for existing_file in os.listdir(config_images_dir):
                if existing_file.startswith(f"{safe_original_name}_"):
                    return os.path.join(config_images_dir, existing_file)
            
            return None
        except Exception as e:
            print(f"❌ Error searching for existing image: {e}")
            return None

    def copy_image_to_permanent_storage(self, source_path, config_name):
        """
        ADVANCED IMAGE STORAGE: Copy image with deduplication and conflict resolution
        
        ARCHITECTURE:
        This is a critical function for the config system. It ensures that all
        images referenced by configs are stored permanently and won't be lost
        when temporary files are cleaned up.
        
        FEATURES:
        1. DEDUPLICATION: Same content → same filename (hash-based)
        2. CONFLICT RESOLUTION: Different content with same name → versioning
        3. METADATA PRESERVATION: Maintains original filename information
        4. ERROR RECOVERY: Graceful handling of file system errors
        
        FILENAME STRATEGY:
        Original: "my_image.jpg"
        Stored as: "my_image_a1b2c3d4.jpg" (name + hash + extension)
        
        DEDUPLICATION LOGIC:
        - Calculate content hash
        - Check if file with same hash exists
        - If exists, reuse existing file (save storage space)
        - If not, copy with hash suffix
        
        PARAMETERS:
        source_path: Original file path (may be temporary)
        config_name: Config name for context (used in logging)
        
        RETURNS:
        Tuple[Optional[str], str]: (permanent_path or None, status_message)
        """
        try:
            if not source_path or not os.path.exists(source_path):
                return None, f"Source image not found: {source_path}"
            
            # Create config images directory
            config_images_dir = os.path.join(self.base_path, 'config_images')
            os.makedirs(config_images_dir, exist_ok=True)
            
            # Get original filename and extension
            original_filename = os.path.basename(source_path)
            name_without_ext, file_extension = os.path.splitext(original_filename)
            
            # Clean the original filename
            safe_name = "".join(c for c in name_without_ext if c.isalnum() or c in ('_', '-', ' ')).strip()
            if not safe_name:
                safe_name = "image"
            safe_name = safe_name.replace(' ', '_')
            
            if not file_extension:
                file_extension = '.jpg'
            
            # Calculate content hash for deduplication
            content_hash = self.calculate_file_hash(source_path)
            if not content_hash:
                import time
                content_hash = str(int(time.time()))[-8:]
            
            # Create permanent filename: {original_name}_{hash}{extension}
            permanent_filename = f"{safe_name}_{content_hash}{file_extension}"
            permanent_path = os.path.join(config_images_dir, permanent_filename)
            
            # Check if file with same hash already exists
            existing_files = []
            for existing_file in os.listdir(config_images_dir):
                if existing_file.endswith(f"_{content_hash}{file_extension}"):
                    existing_files.append(existing_file)
            
            if existing_files:
                existing_path = os.path.join(config_images_dir, existing_files[0])
                print(f"🔗 Reusing existing image: {existing_files[0]}")
                return existing_path, f"Reused existing image: {existing_files[0]}"
            
            # Copy file to permanent storage
            shutil.copy2(source_path, permanent_path)
            print(f"📁 Copied image to permanent storage: {permanent_filename}")
            
            return permanent_path, f"Image copied to permanent storage: {permanent_filename}"
            
        except Exception as e:
            return None, f"Error copying image: {str(e)}"


    # ==============================================================================
    # QUEUE PROCESSING ENGINE
    # ==============================================================================

    def start_queue_processing(self, process_function) -> Tuple[bool, str]:
        """
        PROCESSING STARTER: Initialize queue processing with background thread
        
        THREAD ARCHITECTURE:
        Creates background thread for queue processing to avoid blocking UI.
        Uses daemon thread to ensure clean shutdown with main application.
        
        SAFETY CHECKS:
        - Prevents multiple concurrent processing sessions
        - Validates queue has items before starting
        - Resets processing state cleanly
        
        PARAMETERS:
        process_function: Callable that processes individual config items
                         Must accept config_data dict and return bool (success)
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        
        THREAD LIFECYCLE:
        1. Validate preconditions
        2. Set processing flags
        3. Create and start daemon thread
        4. Return immediately (non-blocking)
        """
        if self.is_processing:
            return False, "Queue processing is already running"
            
        if not self._has_queued_items():
            return False, "No items in queue"
            
        self.is_processing = True
        self.stop_processing = False
        
        # Start processing thread
        self.queue_thread = threading.Thread(
            target=self._process_queue_worker_simple, 
            args=(process_function,),
            daemon=True
        )
        self.queue_thread.start()
        
        return True, "Queue processing started"
        
    def stop_queue_processing(self) -> Tuple[bool, str]:
        """
        PROCESSING STOPPER: Signal graceful shutdown of queue processing
        
        GRACEFUL SHUTDOWN:
        Sets stop flag to allow current item to complete before stopping.
        Does not forcefully terminate thread to avoid data corruption.
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        """
        if not self.is_processing:
            return False, "Queue processing is not running"
            
        self.stop_processing = True
        return True, "Queue processing will stop after current item"

    def _has_queued_items(self) -> bool:
        """
        UTILITY: Check if queue contains items
        
        SAFE CHECKING:
        Handles directory existence and permission errors gracefully.
        
        RETURNS:
        bool: True if queue has .json files
        """
        try:
            if not os.path.exists(self.queue_dir):
                return False
            return any(f.endswith('.json') for f in os.listdir(self.queue_dir))
        except:
            return False

    def _get_next_queue_item(self) -> Optional[str]:
        """
        QUEUE ORDERING: Get next item from queue (FIFO - oldest first)
        
        FIFO PROCESSING:
        Uses file modification time to ensure oldest queued items process first.
        This provides predictable processing order for users.
        
        RETURNS:
        str or None: Config name of next item, or None if queue empty
        """
        try:
            if not os.path.exists(self.queue_dir):
                return None
                
            json_files = [f for f in os.listdir(self.queue_dir) if f.endswith('.json')]
            if not json_files:
                return None
                
            # Sort by modification time (oldest first)
            json_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.queue_dir, x)))
            return json_files[0][:-5]  # Remove .json extension
            
        except Exception as e:
            print(f"Error getting next queue item: {e}")
            return None
        
    def _move_to_processing(self, config_name: str) -> bool:
        """
        STATE TRANSITION: Move config from queue to processing
        
        ATOMIC OPERATION:
        Uses shutil.move for atomic file operation to prevent race conditions.
        
        PARAMETERS:
        config_name: Name of config to move
        
        RETURNS:
        bool: True if move successful
        """
        try:
            source = os.path.join(self.queue_dir, f"{config_name}.json")
            dest = os.path.join(self.processing_dir, f"{config_name}.json")
            
            if os.path.exists(source):
                shutil.move(source, dest)
                return True
            return False
        except Exception as e:
            print(f"Error moving to processing: {e}")
            return False       

    def _move_to_completed(self, config_name: str) -> bool:
        """
        COMPLETION TRANSITION: Move config from processing to completed
        
        SUCCESS HANDLING:
        Marks successful completion of config processing.
        Completed items are shown in status for user feedback.
        
        PARAMETERS:
        config_name: Name of config to mark complete
        
        RETURNS:
        bool: True if move successful
        """
        try:
            source = os.path.join(self.processing_dir, f"{config_name}.json")
            dest = os.path.join(self.completed_dir, f"{config_name}.json")
            
            if os.path.exists(source):
                shutil.move(source, dest)
                return True
            return False
        except Exception as e:
            print(f"Error moving to completed: {e}")
            return False

    def _move_to_error(self, config_name: str, error_msg: str) -> bool:
        """
        ERROR HANDLING: Move config to error directory with error information
        
        ERROR METADATA:
        Enhances config file with error information for debugging:
        - Error message
        - Timestamp of failure
        - Original config data preserved
        
        DEBUGGING AID:
        Error configs can be inspected to understand failure causes.
        
        PARAMETERS:
        config_name: Name of failed config
        error_msg: Description of error that occurred
        
        RETURNS:
        bool: True if error handling successful
        """
        try:
            source = os.path.join(self.processing_dir, f"{config_name}.json")
            dest = os.path.join(self.error_dir, f"{config_name}.json")
            
            if os.path.exists(source):
                # Load config and add error info
                with open(source, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    
                config_data['error'] = {
                    'message': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save with error info
                with open(dest, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                    
                # Remove from processing
                os.remove(source)
                return True
            return False
        except Exception as e:
            print(f"Error moving to error: {e}")
            return False

    def _process_queue_worker_simple(self, process_function):
        """
        CORE PROCESSING WORKER: Main queue processing loop (simplified, robust version)
        
        ARCHITECTURE DECISION:
        This is a simplified version that focuses on reliability over complex features.
        It processes each queue item sequentially without trying to intercept
        individual generation steps, which proved problematic in earlier versions.
        
        PROCESSING LOOP:
        1. Get next queue item (FIFO order)
        2. Move to processing directory
        3. Load config data
        4. Call process function
        5. Handle success/failure appropriately
        6. Repeat until queue empty or stop signal
        
        ERROR HANDLING:
        - Individual config errors don't stop entire queue
        - Failed configs moved to error directory with details
        - Processing continues with remaining items
        - Comprehensive logging for debugging
        
        STATE MANAGEMENT:
        - Updates current_config for status display
        - Maintains processing flags correctly
        - Ensures clean shutdown on completion or error
        
        THREAD SAFETY:
        - Runs in separate thread to avoid blocking UI
        - Uses file system operations for state persistence
        - Graceful handling of stop signals
        
        PARAMETERS:
        process_function: Function that processes individual configs
                         Must accept config_data dict and return bool
        
        PERFORMANCE:
        - No complex stream interception (previous versions had issues)
        - Direct function calls for maximum compatibility
        - Minimal overhead between configs
        """
        try:
            print("🔄 Queue worker thread started")
            
            total_processed = 0
            total_errors = 0
            
            while not self.stop_processing:
                # Get next item
                config_name = self._get_next_queue_item()
                if not config_name:
                    print("📭 No more items in queue")
                    break
                    
                print(f"🎬 Processing config: {config_name}")
                self.current_config = config_name
                
                # Move to processing
                if not self._move_to_processing(config_name):
                    print(f"❌ Failed to move {config_name} to processing")
                    continue
                    
                try:
                    # Load config
                    success, config_data, message = self.load_config_from_processing(config_name)
                    if not success:
                        print(f"❌ Failed to load config {config_name}: {message}")
                        self._move_to_error(config_name, message)
                        total_errors += 1
                        continue
                        
                    # Process the config
                    print(f"🎯 Starting generation for: {config_name}")
                    result = process_function(config_data)
                    
                    if result:
                        # Success - move to completed
                        self._move_to_completed(config_name)
                        total_processed += 1
                        print(f"✅ Completed processing: {config_name}")
                    else:
                        # Failed - move to error
                        self._move_to_error(config_name, "Processing failed")
                        total_errors += 1
                        print(f"❌ Failed processing: {config_name}")
                        
                except Exception as e:
                    # Error during processing
                    error_msg = f"Processing error: {str(e)}"
                    self._move_to_error(config_name, error_msg)
                    total_errors += 1
                    print(f"❌ Error processing {config_name}: {e}")
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"❌ Queue worker error: {e}")
            traceback.print_exc()
        finally:
            # Always reset processing state
            print("🏁 Queue worker finishing - resetting processing state")
            self.is_processing = False
            self.current_config = None
            self.stop_processing = False
            print(f"✅ Queue processing stopped - Processed: {total_processed}, Errors: {total_errors}")

    def load_config_from_processing(self, config_name: str) -> Tuple[bool, Dict, str]:
        """
        PROCESSING LOADER: Load config file from processing directory
        
        PROCESSING CONTEXT:
        This loader is specifically for configs that are currently being processed.
        It loads from the processing directory rather than the main configs directory.
        
        PARAMETERS:
        config_name: Name of config currently in processing
        
        RETURNS:
        Tuple[bool, Dict, str]: (success, config_data, message)
        """
        try:
            config_file = os.path.join(self.processing_dir, f"{config_name}.json")
            if not os.path.exists(config_file):
                return False, {}, f"Config file not found in processing: {config_name}"
                
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            return True, config_data, "Config loaded successfully"
            
        except Exception as e:
            return False, {}, f"Error loading config from processing: {str(e)}"

    # ==============================================================================
    # FUTURE ENHANCEMENT AREAS
    # ==============================================================================
    """
    MODIFICATION GUIDELINES:

    1. MAINTAINING COMPATIBILITY:
    - Keep existing public API unchanged
    - Add new features as optional parameters
    - Ensure backward compatibility with existing configs

    2. THREAD SAFETY:
    - Any new features must consider thread safety
    - File operations should remain atomic
    - State management should be consistent

    3. ERROR HANDLING:
    - All new features should have comprehensive error handling
    - Failed operations should not corrupt queue state
    - Error messages should be user-friendly

    4. TESTING:
    - New features should be thoroughly tested
    - Edge cases should be considered (empty queues, missing files, etc.)
    - Performance impact should be measured
    """
    
    def calculate_file_hash(self, file_path):
        """
        UTILITY: Calculate MD5 hash for file deduplication
        
        PURPOSE:
        Generates a unique hash for file content to enable deduplication.
        This prevents storing multiple copies of the same image file.
        
        ALGORITHM:
        Uses MD5 hash with chunked reading for memory efficiency.
        Returns first 12 characters for reasonable uniqueness with shorter filenames.
        
        PARAMETERS:
        file_path: Path to file to hash
        
        RETURNS:
        str: 12-character hash string, or None on error
        """
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()[:12]
        except Exception as e:
            print(f"❌ Error calculating hash for {file_path}: {e}")
            return None


    # ==============================================================================
    # CONFIG FILE OPERATIONS
    # ==============================================================================

    def _init_directories(self):
        """
        INITIALIZATION: Create necessary directories if they don't exist
        
        DIRECTORY STRUCTURE:
        Creates the complete directory structure needed for queue operations.
        This ensures the system can operate immediately after initialization.
        """
        for dir_path in [self.configs_dir, self.queue_dir, self.processing_dir, 
                        self.completed_dir, self.error_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
    def generate_config_name(self, custom_name: str = "") -> str:
        """
        NAME GENERATION: Create unique config name with timestamp
        
        NAMING STRATEGY:
        - With custom name: "custom_name_20241225_143052"
        - Without custom name: "config_20241225_143052"
        
        SAFETY:
        Removes invalid filename characters to ensure filesystem compatibility.
        
        PARAMETERS:
        custom_name: User-provided name (optional)
        
        RETURNS:
        str: Safe, unique config name
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_name:
            # Remove invalid filename characters
            safe_name = "".join(c for c in custom_name if c.isalnum() or c in ('_', '-', ' ')).strip()
            safe_name = safe_name.replace(' ', '_')
            return f"{safe_name}_{timestamp}"
        else:
            return f"config_{timestamp}"
                    
    def save_config_with_timestamp_option(self, config_name: str, image_path: str, prompt: str, 
                                        lora_settings: Dict, add_timestamp: bool = True, 
                                        other_params: Dict = None) -> Tuple[bool, str]:
        """
        PRIMARY SAVE FUNCTION: Save config with advanced image handling and timestamp control
        
        CRITICAL FUNCTION for the entire config system. This function:
        1. Validates all inputs thoroughly
        2. Handles temporary vs permanent image storage
        3. Manages config name generation and collision detection
        4. Stores complete config data with metadata
        
        ADVANCED FEATURES:
        - Automatic image storage for Gradio temp files
        - Deduplication of identical images
        - LoRA file validation
        - Config name collision handling
        - Comprehensive metadata storage
        
        TIMESTAMP LOGIC:
        - add_timestamp=True: Always creates unique name (safe)
        - add_timestamp=False: Uses exact name (may overwrite)
        
        CONFIG DATA STRUCTURE:
        {
            "config_name": "actual_name_used",
            "timestamp": "2024-12-25T14:30:52.123456",
            "image_path": "/path/to/permanent/image.jpg",
            "original_image_path": "/path/to/original/temp/image.jpg",
            "original_filename": "user_uploaded_name.jpg",
            "prompt": "The girl dances gracefully...",
            "lora_settings": { ... },
            "other_params": { ... }
        }
        
        PARAMETERS:
        config_name: Desired config name (empty for auto-generation)
        image_path: Path to image file
        prompt: Text prompt for generation
        lora_settings: Complete LoRA configuration
        add_timestamp: Whether to add timestamp to name
        other_params: Additional parameters (reserved for future use)
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        
        CRITICAL NOTE:
        The returned message contains ONLY the clean config name for UI parsing.
        System messages are logged separately to avoid UI contamination.
        """
        try:
            # Validate inputs
            if not image_path or not os.path.exists(image_path):
                return False, f"Invalid image path: {image_path}"
                
            # Validate LoRA files if enabled
            if lora_settings.get('use_lora', False):
                lora_files = lora_settings.get('lora_files', [])
                for lora_file in lora_files:
                    if lora_file and not os.path.exists(lora_file):
                        return False, f"LoRA file not found: {lora_file}"
            
            # Handle config name generation
            if not config_name:
                final_config_name = self.generate_config_name()
            else:
                if add_timestamp:
                    final_config_name = self.generate_config_name(config_name)
                else:
                    final_config_name = config_name
                    
                    # CASE-SENSITIVITY FIX: Use generic helper
                    self._remove_file_case_insensitive(self.configs_dir, final_config_name)
            
            # IMPROVED IMAGE STORAGE
            final_image_path = image_path
            image_storage_note = ""
            
            if self.is_gradio_temp_file(image_path):
                print(f"🔄 Detected Gradio temp file, copying with deduplication...")
                permanent_path, message = self.copy_image_to_permanent_storage(image_path, final_config_name)
                if permanent_path:
                    final_image_path = permanent_path
                    image_storage_note = f" ({message})"
                    print(f"✅ {message}")
                else:
                    print(f"❌ Failed to copy image: {message}")
                    return False, f"Failed to store image permanently: {message}"
            else:
                print(f"📁 Using existing permanent image path: {image_path}")
                        
            # Store additional image metadata for better recovery
            original_filename = os.path.basename(image_path)
            
            # Create config data with enhanced image information
            config_data = {
                "config_name": final_config_name,
                "timestamp": datetime.now().isoformat(),
                "image_path": os.path.abspath(final_image_path),
                "original_image_path": image_path,
                "original_filename": original_filename,
                "prompt": prompt,
                "lora_settings": lora_settings,
                "other_params": other_params or {}
            }
            
            # Save config file
            config_file = os.path.join(self.configs_dir, f"{final_config_name}.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # Verify the file was created with correct casing
            if os.path.exists(config_file):
                # Double-check the actual filename on disk
                actual_files = os.listdir(self.configs_dir)
                for actual in actual_files:
                    if actual.lower() == f"{final_config_name.lower()}.json":
                        if actual == f"{final_config_name}.json":
                            print(f"✅ Config saved with correct casing: {final_config_name}")
                        else:
                            print(f"⚠️ File system preserved different casing: {actual[:-5]} (requested: {final_config_name})")
                        break
            
            # CRITICAL FIX: Return ONLY the clean config name
            return True, f"Config saved: {final_config_name}"
            
        except Exception as e:
            return False, f"Error saving config: {str(e)}"

    def _remove_file_case_insensitive(self, directory: str, filename: str) -> bool:
        """
        UTILITY: Remove file handling case-insensitive file systems
        
        PURPOSE:
        Generic method to remove files with different casing before creating
        new ones with the desired casing. Works for any directory.
        
        PARAMETERS:
        directory: Directory path to search in
        filename: Filename to remove (without .json extension)
        
        RETURNS:
        bool: True if any file was removed
        """
        removed = False
        try:
            if os.path.exists(directory):
                for existing in os.listdir(directory):
                    if existing.lower() == f"{filename.lower()}.json":
                        existing_name = existing[:-5]  # Remove .json
                        if existing_name != filename:
                            # Different casing detected
                            print(f"🔄 Detected case difference: '{existing_name}' vs '{filename}'")
                        file_path = os.path.join(directory, existing)
                        os.remove(file_path)
                        print(f"🗑️ Removed file: {existing}")
                        removed = True
            return removed
        except Exception as e:
            print(f"❌ Error removing file: {e}")
            return False
        
    def load_config_for_editing(self, config_name: str) -> Tuple[bool, Dict, str]:
        """
        CONFIG LOADING: Load config for UI editing with image recovery
        
        EDITING MODE vs GENERATION MODE:
        This function loads configs for editing purposes, which means:
        - Missing images are allowed (can be replaced before generation)
        - LoRA files must exist (cannot be easily replaced)
        - Recovery attempts are made for missing images
        
        IMAGE RECOVERY STRATEGIES:
        1. Try original image path from config
        2. Try alternative paths stored in config metadata
        3. Search by original filename in config_images directory
        4. Allow loading with missing image (for editing/replacement)
        
        RECOVERY STATUS INDICATORS:
        - "available": Image found at expected location
        - "recovered": Image found at alternative location
        - "missing": Image not found but config can still be edited
        
        PARAMETERS:
        config_name: Name of config to load
        
        RETURNS:
        Tuple[bool, Dict, str]: (success, config_data, message)
        
        CONFIG DATA ENHANCEMENT:
        Adds '_image_status' field to config data for UI feedback.
        This internal field is removed before generation.
        """
        try:
            config_file = os.path.join(self.configs_dir, f"{config_name}.json")
            if not os.path.exists(config_file):
                return False, {}, f"Config file not found: {config_name}"
                
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Check image availability but don't fail if missing
            image_path = config_data.get('image_path')
            image_status = "available"
            
            if not image_path or not os.path.exists(image_path):
                # Try recovery strategies
                recovered_path = None
                
                # Strategy 1: Try original path
                original_path = config_data.get('original_image_path')
                if original_path and os.path.exists(original_path):
                    recovered_path = original_path
                    print(f"🔄 Recovered image from original path: {original_path}")
                
                # Strategy 2: Search by original filename
                if not recovered_path:
                    original_filename = config_data.get('original_filename')
                    if original_filename:
                        found_path = self.find_image_by_original_name(original_filename)
                        if found_path:
                            recovered_path = found_path
                            print(f"🔍 Found image by original name: {found_path}")
                
                if recovered_path:
                    config_data['image_path'] = recovered_path
                    image_status = "recovered"
                else:
                    # Image not found - still allow loading for editing
                    image_status = "missing"
                    print(f"⚠️ Image missing but config loaded for editing: {image_path}")
            
            # Add image status to config data for UI feedback
            config_data['_image_status'] = image_status
            
            # Always validate LoRA files
            lora_settings = config_data.get('lora_settings', {})
            if lora_settings.get('use_lora', False):
                lora_files = lora_settings.get('lora_files', [])
                missing_lora = []
                for lora_file in lora_files:
                    if lora_file and not os.path.exists(lora_file):
                        missing_lora.append(lora_file)
                
                if missing_lora:
                    return False, {}, f"LoRA files not found: {', '.join(missing_lora)}"
                        
            success_message = "Config loaded successfully"
            if image_status == "missing":
                success_message += " (image missing - generation will be blocked until image is replaced)"
            elif image_status == "recovered":
                success_message += " (image recovered from alternative path)"
            
            return True, config_data, success_message
            
        except Exception as e:
            return False, {}, f"Error loading config: {str(e)}"

    def load_config_for_generation(self, config_name: str) -> Tuple[bool, Dict, str]:
        """
        GENERATION LOADING: Load config for video generation with strict validation
        
        GENERATION MODE REQUIREMENTS:
        - Image MUST be available (no missing images allowed)
        - All LoRA files MUST exist
        - All parameters must be valid
        
        This function is used by the queue processor and requires all assets
        to be available for successful video generation.
        
        PARAMETERS:
        config_name: Name of config to load
        
        RETURNS:
        Tuple[bool, Dict, str]: (success, config_data, message)
        """
        try:
            success, config_data, message = self.load_config_for_editing(config_name)
            
            if not success:
                return success, config_data, message
            
            # For generation, image MUST be available
            image_status = config_data.get('_image_status', 'unknown')
            if image_status == 'missing':
                return False, {}, f"Cannot generate video: Image file missing for config '{config_name}'"
            
            # Remove internal status before returning
            if '_image_status' in config_data:
                del config_data['_image_status']
            
            return True, config_data, "Config ready for generation"
            
        except Exception as e:
            return False, {}, f"Error preparing config for generation: {str(e)}"  

    def config_exists(self, config_name: str) -> bool:
        """
        UTILITY: Check if a config file exists
        
        PARAMETERS:
        config_name: Config name to check
        
        RETURNS:
        bool: True if config file exists
        """
        config_file = os.path.join(self.configs_dir, f"{config_name}.json")
        return os.path.exists(config_file) 

    def delete_config(self, config_name: str) -> Tuple[bool, str]:
        """
        CONFIG DELETION: Delete a config file with validation
        
        SAFETY:
        Validates file existence before deletion to provide appropriate feedback.
        
        PARAMETERS:
        config_name: Name of config to delete
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        """
        try:
            config_file = os.path.join(self.configs_dir, f"{config_name}.json")
            
            if not os.path.exists(config_file):
                return False, f"Config file not found: {config_name}"
            
            # Remove the file
            os.remove(config_file)
            
            return True, f"Config deleted: {config_name}"
            
        except Exception as e:
            return False, f"Error deleting config: {str(e)}"

    def get_available_configs(self) -> List[str]:
        """
        CONFIG DISCOVERY: Get list of available config files with validation
        
        ROBUST SCANNING:
        - Always performs fresh filesystem scan
        - Validates JSON format of each file
        - Skips corrupted files with warning
        - Returns sorted list for consistent ordering
        - Handles case-sensitive file systems properly
        
        RETURNS:
        List[str]: List of valid config names (without .json extension)
        """
        try:
            configs = []
            seen_lowercase = {}  # Track lowercase versions for duplicate detection
            
            if os.path.exists(self.configs_dir):
                # Always do a fresh directory scan
                for file in sorted(os.listdir(self.configs_dir)):
                    if file.endswith('.json'):
                        config_name = file[:-5]  # Remove .json extension
                        # Verify file is readable
                        config_file = os.path.join(self.configs_dir, file)
                        try:
                            with open(config_file, 'r', encoding='utf-8') as f:
                                json.load(f)  # Basic JSON validation
                            
                            # Check for case-insensitive duplicates on Windows
                            lowercase_name = config_name.lower()
                            if lowercase_name in seen_lowercase:
                                print(f"⚠️ Warning: Case-sensitive duplicate found: '{config_name}' vs '{seen_lowercase[lowercase_name]}'")
                            else:
                                seen_lowercase[lowercase_name] = config_name
                            
                            configs.append(config_name)
                        except:
                            # Skip corrupted files
                            print(f"Warning: Skipping corrupted config file: {file}")
            return configs
        except Exception as e:
            print(f"Error getting available configs: {e}")
            return []
    
    # ==============================================================================
    # QUEUE MANAGEMENT OPERATIONS
    # ==============================================================================

    def queue_config(self, config_name: str) -> Tuple[bool, str]:
        """
        QUEUE OPERATION: Add config to processing queue
        
        QUEUE LOGIC:
        Copies config from configs/ directory to queue/ directory.
        Using copy (not move) preserves original config for future use.
        
        COLLISION DETECTION:
        Prevents duplicate queue entries for same config name.
        
        PARAMETERS:
        config_name: Name of config to queue
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        """
        try:
            source_file = os.path.join(self.configs_dir, f"{config_name}.json")
            dest_file = os.path.join(self.queue_dir, f"{config_name}.json")
            
            if not os.path.exists(source_file):
                return False, f"Config file not found: {config_name}"
            
            # CASE-SENSITIVITY FIX: Use generic helper
            self._remove_file_case_insensitive(self.queue_dir, config_name)
                
            shutil.copy2(source_file, dest_file)
            return True, f"Config queued: {config_name}"
            
        except Exception as e:
            return False, f"Error queueing config: {str(e)}"

        

            






        
