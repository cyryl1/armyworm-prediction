import os
import httpx
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class APIClient:
    """Async client for EPPO API with retries and circuit breaker."""
    
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.timeout = float(os.getenv("LIVE_API_TIMEOUT_SECONDS", "3.0"))
        self.eppo_token = os.getenv("EPPO_AUTH_TOKEN")
        
        # Circuit breaker state
        self.consecutive_failures = 0
        self.circuit_open_until: Optional[float] = None
        self.CIRCUIT_BREAKER_THRESHOLD = 3
        self.CIRCUIT_BREAKER_RESET_SECONDS = 300  # 5 minutes
        
    def init_client(self):
        """Initialize the httpx client (call during startup)."""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout)
            
    async def close(self):
        """Close the httpx client."""
        if self.client:
            await self.client.aclose()
            self.client = None
            
    def _check_circuit_breaker(self) -> bool:
        """Return True if we should attempt the API call, False if circuit is open."""
        if self.circuit_open_until:
            if time.time() > self.circuit_open_until:
                # Reset breaker
                self.circuit_open_until = None
                self.consecutive_failures = 0
                return True
            return False
        return True
        
    def _record_failure(self):
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.CIRCUIT_BREAKER_THRESHOLD:
            self.circuit_open_until = time.time() + self.CIRCUIT_BREAKER_RESET_SECONDS
            logger.warning("API circuit breaker tripped. Disabling live API for 5 minutes.")

    def _record_success(self):
        self.consecutive_failures = 0
        self.circuit_open_until = None
            
    async def fetch_eppo_data(self, eppo_code: str) -> Optional[Dict[str, Any]]:
        """Fetch taxon, distribution, and BCA data from EPPO API v2."""
        if not self.eppo_token or not self._check_circuit_breaker() or not self.client:
            return None
            
        headers = {"X-Api-Key": self.eppo_token}
        base_url = "https://api.eppo.int/gd/v2/taxons/taxon"
        
        try:
            # We'll fetch overview and BCAs concurrently to save time
            # EPPO API V2 limits are generous (2000 per 10s)
            overview_task = self.client.get(f"{base_url}/{eppo_code}/overview", headers=headers)
            bca_task = self.client.get(f"{base_url}/{eppo_code}/bca", headers=headers)
            
            responses = await asyncio.gather(overview_task, bca_task, return_exceptions=True)
            
            overview_res, bca_res = responses
            
            # Check for HTTP errors
            if isinstance(overview_res, Exception) or isinstance(bca_res, Exception):
                raise Exception("Network error during asyncio.gather")
                
            if overview_res.status_code != 200:
                logger.warning(f"EPPO overview API returned {overview_res.status_code}")
                self._record_failure()
                return None
                
            self._record_success()
            
            data = {"overview": overview_res.json()}
            
            if bca_res.status_code == 200:
                data["bca"] = bca_res.json()
            else:
                data["bca"] = []
                
            return data
            
        except httpx.TimeoutException:
            logger.warning(f"EPPO API timeout for {eppo_code}")
            self._record_failure()
            return None
        except Exception as e:
            logger.error(f"EPPO API error for {eppo_code}: {e}")
            self._record_failure()
            return None

# Global API client instance
api_client = APIClient()
