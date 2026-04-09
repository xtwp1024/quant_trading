import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - SENTIMENT - %(levelname)s - %(message)s')
logger = logging.getLogger("SentimentEngine")

class SentimentEngine:
    def __init__(self, config_path="config.yaml"):
        import yaml
        self.use_mock = True
        self.api = None
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            
            tw_conf = self.config.get('twitter_api', {})
            if tw_conf.get('enabled'):
                try:
                    import tweepy
                    auth = tweepy.OAuthHandler(tw_conf['api_key'], tw_conf['api_secret'])
                    auth.set_access_token(tw_conf['access_token'], tw_conf['access_secret'])
                    self.api = tweepy.API(auth)
                    self.use_mock = False
                    logger.info("✅ Connected to Real X API")
                except Exception as e:
                    logger.error(f"❌ Failed to connect to X API: {e}. Reverting to Mock.")
                    self.use_mock = True
        except Exception:
            logger.warning("Config load failed or missing. Using Mock.")

    def analyze(self, symbol="ETH"):
        """
        Scans Social Media (Real or Mock).
        Returns -1.0 to 1.0.
        """
        if self.use_mock:
            return self._analyze_mock(symbol)
        else:
            return self._analyze_real(symbol)

    def _analyze_real(self, symbol):
        try:
            # Simple keyword search
            query = f"{symbol} OR #Ethereum -filter:retweets" 
            tweets = self.api.search_tweets(q=query, count=20, lang="en")
            
            if not tweets: return {"score": 0.0, "sources_count": 0, "top_keywords": []}
            
            # Simple NLP (Keyword Bag)
            bullish_words = ["bull", "moon", "up", "long", "buy", "breakout", "green"]
            bearish_words = ["bear", "dump", "down", "short", "sell", "crash", "red"]
            
            score = 0
            keywords = []
            
            for t in tweets:
                text = t.text.lower()
                for w in bullish_words:
                    if w in text: 
                        score += 1
                        keywords.append(w)
                for w in bearish_words: 
                    if w in text: 
                        score -= 1
                        keywords.append(w)
            
            # Normalize -1.0 to 1.0 based on volume
            final_score = score / (len(tweets) + 1) # Simple normalization
            final_score = max(-1.0, min(1.0, final_score))
            
            return {
                "score": final_score,
                "sources_count": len(tweets),
                "top_keywords": list(set(keywords))[:5]
            }
        except Exception as e:
            logger.error(f"Real analysis failed: {e}")
            return self._analyze_mock(symbol)

    def _analyze_mock(self, symbol):
        # MOCK IMPLEMENTATION
        # Simulate scanning
        sources = ["X (Mock)", "Reddit (Mock)", "Bloomberg (Mock)"]
        
        # Random bias based on "Trend" simulation
        # Let's bias it slightly positive for the bull market vibe
        base_sentiment = random.gauss(0.1, 0.4) 
        
        # Clamp
        final_sentiment = max(-1.0, min(1.0, base_sentiment))
        
        logger.info(f"Analyzing {symbol} Sentiment via {sources}...")
        logger.info(f"📊 Sentiment Score: {final_sentiment:.2f}")
        
        return {
            "score": final_sentiment,
            "sources_count": random.randint(50, 500),
            "top_keywords": ["bullish", "upgrade", "liquidations"] if final_sentiment > 0 else ["fud", "regulation", "sell"]
        }
