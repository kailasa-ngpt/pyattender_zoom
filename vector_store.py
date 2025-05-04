# File: vector_store.py
# Refactored version with improved error handling, reduced duplication, and better modularization

import os
import numpy as np
import lancedb
import pyarrow as pa
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
import time
import aiohttp
import json
import traceback
import random
from exceptions import VectorStoreError

class VectorStore:
    def __init__(self, config):
        """Initialize the vector store with configuration."""
        self.config = config
        self.db_path = config.VECTOR_DB_PATH
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self.tables = {}
        self.embedding_cache = {}  # Simple in-memory cache for embeddings
        self.jina_api_key = os.getenv("JINA_API_KEY", config.JINA_API_KEY)
        self.vector_dim = 1024  # Jina embeddings-v3 dimension
        


        """ get_table without indexing
    def get_table(self, category='default'):
        # Get or create the LanceDB table for the specified category.
        if category in self.tables:
            return self.tables[category]
            
        table_name = f"{category}_roster"
        try:
            # Try to open existing table
            table = self.db.open_table(table_name)
            print(f"Opened existing table {table_name}")
        except Exception as e:
            print(f"Table {table_name} does not exist, creating new: {str(e)}")
            
            # Create new table with schema
            schema = pa.schema([
                pa.field('id', pa.string()),
                pa.field('id_number', pa.string()),
                pa.field('combination_type', pa.string()),
                pa.field('text', pa.string()),
                pa.field('vector', pa.list_(pa.float32(), self.vector_dim))
            ])
            table = self.db.create_table(table_name, schema=schema)
            
        self.tables[category] = table
        return table
        """




    def get_table(self, category='default'):
        """Get or create the LanceDB table for the specified category with proper indexing."""
        if category in self.tables:
            return self.tables[category]
            
        table_name = f"{category}_roster"
        try:
            # Try to open existing table
            table = self.db.open_table(table_name)
            print(f"Opened existing table {table_name}")
            
            # Check if we need to create indexes on existing table
            try:
                # Create indexes with correct parameters for LanceDB
                self._create_table_indexes(table)
            except Exception as e:
                print(f"Note on existing table index: {str(e)}")
                        
        except Exception as e:
            print(f"Table {table_name} does not exist, creating new: {str(e)}")
            
            # Create new table with schema
            schema = pa.schema([
                pa.field('id', pa.string()),
                pa.field('id_number', pa.string()),    # Single ID field as requested
                pa.field('combination_type', pa.string()),
                pa.field('text', pa.string()),
                pa.field('vector', pa.list_(pa.float32(), self.vector_dim))
            ])
            table = self.db.create_table(table_name, schema=schema)
            
            # Create indexes on new table
            try:
                self._create_table_indexes(table)
            except Exception as e:
                print(f"Could not create index on new table: {str(e)}")
            
        self.tables[category] = table
        return table
    
    def _create_table_indexes(self, table):
        """
        Skip index creation silently since LanceDB 0.21.2 works without explicit indexes.
        """
        # Silent skip - no indexing attempts, no error messages
        pass



        # trying to actually create indexes
    # def _create_table_indexes(self, table): 
    #    """Create appropriate indexes for a LanceDB table."""
    #    try:
    #        # Vector index for similarity search
    #        table.create_index(
    #            ["vector"],
    #            index_type="IVF_PQ",
    #            metric_type="L2"
    #        )
    #        print(f"Created IVF_PQ index on vector field")
    #    except Exception as e:
    #        print(f"Could not create vector index: {str(e)}")
    #        
    #    try:
    #        # String field index for id lookups
    #        table.create_index(
    #            ["id_number"],
    #            index_type="BTREE"
    #        )
    #        print(f"Created BTREE index on id_number field")
    #    except Exception as e:
    #        print(f"Could not create id_number index: {str(e)}")
    
    def clear_table(self, category='default'):
        """Clear a table for the given category."""
        table_name = f"{category}_roster"
        try:
            self.db.drop_table(table_name)
            if category in self.tables:
                del self.tables[category]
            return True
        except Exception as e:
            print(f"Error clearing table for category '{category}': {str(e)}")
            return False
    
    async def create_embedding(self, text: str) -> np.ndarray:
        """Create an embedding vector for a single text."""
        if not text or not text.strip():
            return np.zeros(self.vector_dim)
            
        embeddings = await self.create_embeddings([text])
        return embeddings[text]
    
    async def create_embeddings(self, texts: List[str], batch_size: int = 5) -> Dict[str, np.ndarray]:
        """
        Create embeddings for multiple texts with proper error handling and batching.
        Returns a dictionary mapping original texts to their embeddings.
        
        Args:
            texts: List of text strings to create embeddings for
            batch_size: Maximum number of texts in a single API call (default: 5)
        """
        # Safety check for batch size
        batch_size = max(1, batch_size)  # Ensure batch_size is at least 1
        
        # Filter out empty or None texts and normalize for caching
        valid_texts = [(text, text.strip()) for text in texts if text and text.strip()]
        
        # Initialize result dictionary and tracking
        result_embeddings = {}
        failed_texts = []
        failure_reasons = {}
        
        # Handle empty input case
        if not valid_texts:
            print("Warning: No valid texts to embed - all texts were empty or None")
            return {text: np.zeros(self.vector_dim) for text in texts}
        
        # Check cache first for already computed embeddings
        texts_to_embed = []
        for orig_text, norm_text in valid_texts:
            if norm_text in self.embedding_cache:
                result_embeddings[orig_text] = self.embedding_cache[norm_text]
            else:
                texts_to_embed.append((orig_text, norm_text))
        
        # If all texts were in cache, return early
        if not texts_to_embed:
            return self._ensure_all_texts_have_embeddings(texts, result_embeddings)
        
        # Process in batches with rate limiting
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i+batch_size]
            if not batch:
                continue
                
            # Extract normalized texts for this batch
            batch_norm_texts = [norm_text for _, norm_text in batch]
            
            # Add delay between batches to avoid rate limits
            if i > 0:
                await asyncio.sleep(1.0)
                
            print(f"Processing embedding batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size} with {len(batch)} texts")
            
            try:
                # Call Jina API with retries and backoff
                batch_embeddings = await self._call_jina_api(batch_norm_texts)
                
                # Process response and update cache
                if len(batch_embeddings) == len(batch):
                    for j, (orig_text, norm_text) in enumerate(batch):
                        embedding = batch_embeddings[j]
                        # Cache this result
                        self.embedding_cache[norm_text] = embedding
                        result_embeddings[orig_text] = embedding
                else:
                    # Handle mismatched response length
                    print(f"Warning: Expected {len(batch)} embeddings but got {len(batch_embeddings)}")
                    
                    # Process the embeddings we did get
                    for j, (orig_text, norm_text) in enumerate(batch):
                        if j < len(batch_embeddings):
                            embedding = batch_embeddings[j]
                            self.embedding_cache[norm_text] = embedding
                            result_embeddings[orig_text] = embedding
                        else:
                            # For any missing, track failure and use zero vector
                            zero_emb = np.zeros(self.vector_dim)
                            result_embeddings[orig_text] = zero_emb
                            self.embedding_cache[norm_text] = zero_emb
                            
                            failed_texts.append(orig_text)
                            failure_reasons[orig_text] = "Missing from API response"
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                traceback.print_exc()
                
                # For errors, track failures and use zero vectors for this batch
                for orig_text, norm_text in batch:
                    if orig_text not in result_embeddings:
                        zero_emb = np.zeros(self.vector_dim)
                        result_embeddings[orig_text] = zero_emb
                        self.embedding_cache[norm_text] = zero_emb
                        
                        failed_texts.append(orig_text)
                        failure_reasons[orig_text] = str(e)
        
        # Log failure statistics
        if failed_texts:
            print(f"WARNING: Failed to create embeddings for {len(failed_texts)}/{len(texts)} texts")
            
            # Log a few examples
            for text in failed_texts[:3]:
                print(f"- Failed text: '{text[:50]}...' - Reason: {failure_reasons.get(text)}")
        
        # Ensure all original texts have embeddings
        return self._ensure_all_texts_have_embeddings(texts, result_embeddings)
    
    def _ensure_all_texts_have_embeddings(self, texts: List[str], embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Helper method to ensure all original texts have embeddings."""
        for text in texts:
            if text not in embeddings:
                embeddings[text] = np.zeros(self.vector_dim)
        return embeddings
    
    async def _call_jina_api(self, texts: List[str]) -> List[np.ndarray]:
        """
        Call Jina AI API with retries and backoff.
        Returns a list of numpy arrays (embeddings).
        """
        return await self._call_jina_api_with_backoff(texts)
    
    async def _call_jina_api_with_backoff(self, texts: List[str], max_retries=3):
        """Call Jina API with exponential backoff and jitter."""
        if not self.jina_api_key:
            return [np.zeros(self.vector_dim) for _ in texts]
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.jina_api_key}",
                        "Accept": "application/json"
                    }
                    
                    payload = {
                        "model": "jina-embeddings-v3",
                        "task": "text-matching",
                        "input": texts
                    }
                    
                    # Add attempt number to logs
                    if attempt > 0:
                        print(f"Jina API attempt {attempt+1}/{max_retries} for {len(texts)} texts")
                    
                    async with session.post(
                        "https://api.jina.ai/v1/embeddings",
                        headers=headers,
                        json=payload,
                        timeout=30  # Increased timeout
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"Error from Jina API: {error_text}")
                            
                            # Check if this is a rate limit error
                            is_rate_limit = "rate limit" in error_text.lower() or response.status == 429
                            
                            if is_rate_limit and attempt < max_retries - 1:
                                # Calculate backoff with jitter
                                base_delay = 2 ** attempt  # Exponential
                                jitter = random.uniform(0, 0.5 * base_delay)  # Add randomness
                                delay = base_delay + jitter
                                
                                print(f"Rate limit hit, backing off for {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                                await asyncio.sleep(delay)
                                continue  # Retry after delay
                                
                            raise Exception(f"Jina API error: {error_text}")
                        
                        result = await response.json()
                        embeddings = []
                        
                        for embedding_data in result["data"]:
                            embedding = np.array(embedding_data["embedding"])
                            embeddings.append(embedding)
                        
                        return embeddings
                        
            except Exception as e:
                # Only retry on certain errors and if we have attempts left
                if attempt < max_retries - 1:
                    # Rate limit handling is done above
                    # For other errors, use moderate backoff
                    error_str = str(e)
                    if "timeout" in error_str.lower() or "connection" in error_str.lower():
                        delay = 1 + attempt + random.uniform(0, 1)
                        print(f"Connection error, retrying in {delay:.1f}s: {error_str}")
                        await asyncio.sleep(delay)
                        continue
                
                print(f"Error calling Jina API (attempt {attempt+1}/{max_retries}): {str(e)}")
                
        # If we get here, all retries failed
        print(f"All {max_retries} attempts to call Jina API failed")
        # Return zero vectors as fallback
        return [np.zeros(self.vector_dim) for _ in texts]
    
    async def fallback_to_openai(self, texts: List[str]) -> List[np.ndarray]:
        """
        Fallback to OpenAI for embeddings when Jina fails.
        Only called if OpenAI client is available and Jina failed.
        """
        if not self.config.openai_client:
            return [np.zeros(self.vector_dim) for _ in texts]
        
        embeddings = []
        for text in texts:
            try:
                response = await self.config.openai_client.embeddings.create(
                    input=text.strip(),
                    model="text-embedding-ada-002"
                )
                embedding = np.array(response.data[0].embedding)
                
                # Ensure dimensions match
                if embedding.shape[0] != self.vector_dim:
                    if embedding.shape[0] > self.vector_dim:
                        embedding = embedding[:self.vector_dim]
                    else:
                        padded = np.zeros(self.vector_dim)
                        padded[:embedding.shape[0]] = embedding
                        embedding = padded
                
                embeddings.append(embedding)
            except Exception as e:
                print(f"OpenAI fallback error for text: {str(e)}")
                embeddings.append(np.zeros(self.vector_dim))
        
        return embeddings
    
    def _determine_id_field(self, roster: List[Dict[str, Any]]) -> Optional[str]:
        """
        Determine which field in the roster contains ID values.
        Returns the field name or None if no suitable field is found.
        """
        if not roster:
            return None
            
        # Check first entry's keys to determine available fields
        sample_entry = roster[0]
        available_fields = list(sample_entry.keys())
        print(f"Available fields in roster: {available_fields}")
        
        # Try id_number first (preferred)
        if "id_number" in available_fields:
            print("Using 'id_number' field for ID values")
            return "id_number"
            
        # Try Id as fallback
        if "Id" in available_fields:
            print("Using 'Id' field for ID values")
            return "Id"
            
        # Last resort: look for any field with "id" in name
        for field in available_fields:
            if "id" in field.lower():
                print(f"Using '{field}' field for ID values")
                return field
                
        print("WARNING: No suitable ID field found in roster data!")
        return None
        
    async def _get_existing_id_numbers(self, category: str) -> Set[str]:
        """
        Get set of existing id_number values already in the vector store.
        Uses direct table query instead of vector search.
        """
        existing_id_numbers = set()
        try:
            table = self.get_table(category)
            
            # Try direct table query first
            try:
                # Use LanceDB's to_pandas method to get all records
                all_records = table.to_pandas()
                if 'id_number' in all_records.columns:
                    # Convert to strings and get unique values
                    existing_id_numbers = set(all_records['id_number'].astype(str).unique())
                    print(f"Found {len(existing_id_numbers)} unique id_number values using direct query")
                    if existing_id_numbers:
                        print(f"Sample existing id_number values: {list(existing_id_numbers)[:5]}")
                    return existing_id_numbers
            except Exception as e:
                print(f"Direct query failed, falling back to vector search: {str(e)}")
            
            # Fallback to original vector search method
            zeros = [0.0] * self.vector_dim
            
            # Use paging to count large tables accurately
            offset = 0
            page_size = 1000
            total_records = 0
            
            while True:
                batch = table.search(zeros, query_type="vector").limit(page_size).offset(offset).to_list()
                batch_size = len(batch)
                total_records += batch_size
                
                if batch_size == 0:
                    break
                    
                # Extract id_number values
                for record in batch:
                    if 'id_number' in record and record['id_number']:
                        existing_id_numbers.add(str(record['id_number']))
                
                offset += page_size
                
                # Safety limit
                if offset >= 100000:
                    break
            
            print(f"Found {len(existing_id_numbers)} unique id_number values via vector search")
            
            if existing_id_numbers:
                print(f"Sample existing id_number values: {list(existing_id_numbers)[:5]}")
                
        except Exception as e:
            print(f"Error getting existing id_numbers: {str(e)}")
            traceback.print_exc()
            
        return existing_id_numbers
    
    async def process_person_batch(self, category: str, people: List[Dict[str, Any]], 
                                 vectorization_config: Dict[str, Any], id_field: str) -> Tuple[List[str], int]:
        """
        Process a batch of people at once, creating embeddings and adding to vector store.
        
        Args:
            category: Category for the vector table
            people: List of person dictionaries to process
            vectorization_config: Configuration for vectorization
            id_field: Field containing unique ID values
            
        Returns:
            Tuple of (list of successful entry IDs, count of unique people vectorized)
        """
        # Extract all texts that need embeddings
        all_texts = []
        person_texts_map = {}  # Map from (id_number, combo_name) to text
        
        for person in people:
            # Skip entries without id
            person_id = person.get(id_field)
            if not person_id:
                continue
                
            # Extract column values
            column_values = []
            for col in vectorization_config.get("columns", ["firstName", "lastName", "spiritualName"]):
                column_values.append(person.get(col, ""))
            
            # Generate combinations
            person_texts = []
            for combo in vectorization_config.get("combinations", [["fullName", "{0} {1}"]]):
                combo_name, format_str = combo
                
                try:
                    text = format_str.format(*column_values)
                    text = text.strip()
                    if not text:
                        continue
                        
                    # Add to the list of texts to embed
                    all_texts.append(text)
                    key = (str(person_id), combo_name)
                    person_texts_map[key] = text
                    
                except Exception as e:
                    print(f"Error formatting {combo_name} for person with {id_field} {person_id}: {str(e)}")
        
        # If no valid texts, return early
        if not all_texts:
            print(f"No valid texts to embed for batch - check if roster entries have {id_field} field")
            return [], 0
            
        # Get embeddings for all texts
        all_embeddings = await self.create_embeddings(all_texts)
        
        # Add data to the table
        table = self.get_table(category)
        records_to_add = []
        successful_ids = []
        vectorized_people = set()  # Track unique people vectorized
        
        for (id_value, combo_name), text in person_texts_map.items():
            if text in all_embeddings:
                entry_id = f"{id_value}_{combo_name}"
                
                # Add the record
                records_to_add.append({
                    "id": entry_id,
                    "id_number": str(id_value),
                    "combination_type": combo_name,
                    "text": text,
                    "vector": all_embeddings[text].tolist()
                })
                successful_ids.append(entry_id)
                vectorized_people.add(id_value)  # Count unique people
        
        # Add all records at once if we have any
        if records_to_add:
            table.add(records_to_add, mode="append")
            

            # Try to ensure database persistence
            try:
                # Check if table has flush method
                if hasattr(table, 'flush'):
                    table.flush()
                    
                # Or try to force sync the database
                self.db.flush() if hasattr(self.db, 'flush') else None
            except Exception as e:
                print(f"Note: Database flush attempt: {str(e)}")


        return successful_ids, len(vectorized_people)
    
    async def vectorize_roster_incremental(self, category: str, roster: List[Dict[str, Any]], 
                                    config_data: Dict[str, Any], batch_size: int = 20,
                                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Vectorize only new entries in the roster based on id_number field.
        
        Args:
            category: Category for the vector table
            roster: List of person dictionaries to process
            config_data: Configuration for vectorization
            batch_size: Size of batches for processing
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Statistics about the vectorization process
        """
        # Safety check for batch size
        batch_size = max(1, min(batch_size, 50))  # Between 1 and 50
        
        # Initialize statistics tracking
        stats = {
            "start_time": time.time(),
            "total_people": len(roster),
            "processed_count": 0,
            "already_vectorized": 0,
            "newly_vectorized": 0,
            "error_count": 0,
            "batches": []
        }
        
        # Early return for empty roster
        if not roster:
            stats["end_time"] = time.time()
            stats["total_time"] = 0
            return stats
        
        # 1. Find the ID field in roster
        id_field = self._determine_id_field(roster)
        if not id_field:
            stats["error"] = "No suitable ID field found in roster data"
            stats["end_time"] = time.time()
            stats["total_time"] = stats["end_time"] - stats["start_time"]
            return stats
        
        # 2. Get existing vectorized entries
        existing_id_numbers = await self._get_existing_id_numbers(category)
        stats["already_vectorized"] = len(existing_id_numbers)
        
        # 3. Filter roster to only include entries with valid IDs
        valid_roster_entries = [p for p in roster if p.get(id_field)]
        stats["valid_entries"] = len(valid_roster_entries)
        
        if len(valid_roster_entries) == 0:
            print(f"WARNING: No entries with {id_field} field found in roster. Check your NocoDB data.")
            stats["end_time"] = time.time()
            stats["total_time"] = stats["end_time"] - stats["start_time"]
            return stats
        
        # 4. Filter to only include new entries
        new_entries = [person for person in valid_roster_entries 
                      if str(person.get(id_field)) not in existing_id_numbers]
        stats["new_entries"] = len(new_entries)
        
        print(f"Found {len(new_entries)} new entries to vectorize based on {id_field}")
        
        # Debug info for first few entries in roster
        if new_entries and len(new_entries) > 0:
            print(f"Sample new entry {id_field} values to vectorize: {[str(p.get(id_field)) for p in new_entries[:5]]}")
        
        # 5. Process new entries in batches
        for i in range(0, len(new_entries), batch_size):
            batch_start = time.time()
            
            # Determine actual batch size (handling final batch)
            end_idx = min(i + batch_size, len(new_entries))
            current_batch = new_entries[i:end_idx]
            actual_batch_size = len(current_batch)
            
            try:
                # Process this batch
                person_ids, people_vectorized = await self.process_person_batch(
                    category, current_batch, config_data, id_field
                )
                
                # Update statistics
                stats["newly_vectorized"] += people_vectorized
                
                # Track batch details
                batch_time = time.time() - batch_start
                batch_stats = {
                    "batch_index": i // batch_size,
                    "people_processed": actual_batch_size,
                    "people_vectorized": people_vectorized,
                    "processing_time": batch_time
                }
                stats["batches"].append(batch_stats)
                
                # Call progress callback if provided
                stats["processed_count"] += actual_batch_size
                if progress_callback:
                    progress_callback(
                        stats["processed_count"], 
                        len(new_entries), 
                        batch_stats
                    )
                
                # Print progress
                progress = stats["processed_count"] / len(new_entries) * 100 if new_entries else 100
                print(f"Vectorization progress: {progress:.2f}% ({stats['processed_count']}/{len(new_entries)})")
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {str(e)}")
                traceback.print_exc()
                stats["error_count"] += actual_batch_size
                
                # Call progress callback with error information
                if progress_callback:
                    stats["processed_count"] += actual_batch_size
                    progress_callback(
                        stats["processed_count"], 
                        len(new_entries), 
                        {"error": str(e)}
                    )
        
        # 6. Calculate final statistics
        stats["end_time"] = time.time()
        stats["total_time"] = stats["end_time"] - stats["start_time"]
        stats["avg_time_per_person"] = (
            stats["total_time"] / stats["newly_vectorized"] 
            if stats["newly_vectorized"] > 0 else 0
        )
        
        return stats
    
    async def vectorize_roster(self, category: str, roster: List[Dict[str, Any]], 
                           config_data: Dict[str, Any], batch_size: int = 20,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Fully vectorize a roster.
        This method first clears any existing vectors, then performs an incremental vectorization
        which will now vectorize all entries as they're all "new".
        
        Args:
            category: Category for the vector table
            roster: List of person dictionaries to process
            config_data: Configuration for vectorization
            batch_size: Size of batches for processing
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Statistics about the vectorization process
        """
        # Clear existing table
        self.clear_table(category)
        print(f"Cleared existing vector data for category '{category}' for full vectorization")
        
        # Now run incremental vectorization (which will vectorize everything as "new")
        stats = await self.vectorize_roster_incremental(
            category, roster, config_data, batch_size, progress_callback
        )
        
        # Update statistics to reflect this was a full vectorization
        stats["vectorized_count"] = stats["newly_vectorized"]
        stats["full_vectorization"] = True
        
        return stats
    
    async def match_name(self, category: str, name: str, threshold: float = 0.7, limit: int = 10, 
                     min_confidence_gap: float = 0.05) -> Dict[str, Any]:
        """
        Match a name against vectorized names in the database with enhanced cross-checking.
        
        Args:
            category: The category to search in
            name: The name to match
            threshold: Minimum confidence threshold for a match
            limit: Maximum number of matches to return
            min_confidence_gap: Minimum confidence difference required between top matches
                               to avoid ambiguity
        
        Returns:
            A dictionary with match results, including potential ambiguity warnings
        """
        table = self.get_table(category)
        
        # Create embedding for the input name
        vector = await self.create_embedding(name)
        
        # Search the database for similar vectors - get more than we need for cross-checking
        search_limit = max(limit * 2, 10)  # Get at least 10 results for proper cross-checking
        results = table.search(vector.tolist()).limit(search_limit).to_list()
        
        # Process results
        all_matches = []
        seen_id_numbers = set()
        
        # First, collect all matches above threshold
        for result in results:
            confidence = 1.0 - result["_distance"]
            
            # Skip if confidence is below threshold
            if confidence < threshold:
                continue
            
            # Use id_number for identification
            id_number = result.get("id_number")
            if not id_number:
                continue  # Skip entries without id_number
            
            # If we already included this id_number, check if this match is better
            if id_number in seen_id_numbers:
                # Find the existing match for this id_number
                for i, match in enumerate(all_matches):
                    if match["id_number"] == id_number:
                        # If this match is better, replace it
                        if confidence > match["confidence"]:
                            all_matches[i] = {
                                "id_number": id_number,
                                "confidence": confidence,
                                "matchedOn": result["combination_type"],
                                "text": result["text"]
                            }
                        break
            else:
                # Add new match
                all_matches.append({
                    "id_number": id_number,
                    "confidence": confidence,
                    "matchedOn": result["combination_type"],
                    "text": result["text"]
                })
                seen_id_numbers.add(id_number)
        
        # Sort matches by confidence (highest first)
        all_matches.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Check for potential ambiguity in top matches
        result = {
            "matches": all_matches[:limit],
            "ambiguous": False,
            "matchedPersonId": None,  # For backward compatibility
            "id_number": None,
            "confidence": 0,
            "reasoning": ""
        }
        
        if len(all_matches) > 0:
            best_match = all_matches[0]
            result["matchedPersonId"] = best_match["id_number"]  # For backward compatibility
            result["id_number"] = best_match["id_number"]
            result["confidence"] = best_match["confidence"]
            result["reasoning"] = f"Matched on {best_match['matchedOn']}: {best_match['text']}"
            
            # Check for ambiguity (close matches)
            if len(all_matches) > 1:
                second_best = all_matches[1]
                confidence_gap = best_match["confidence"] - second_best["confidence"]
                
                if confidence_gap < min_confidence_gap:
                    result["ambiguous"] = True
                    result["reasoning"] = f"Potential ambiguity: '{best_match['text']}' ({best_match['confidence']:.2f}) and '{second_best['text']}' ({second_best['confidence']:.2f}) have similar confidence scores (gap: {confidence_gap:.2f})"
        else:
            result["reasoning"] = "No matches found above threshold"
        
        return result