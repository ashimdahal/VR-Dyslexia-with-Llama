import cv2
import numpy as np
import os
from PIL import Image

from paddleocr import PaddleOCR
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import llama_cpp
from llama_cpp import Llama

from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Dict

import logging
import time
from dataclasses import dataclass
from collections import deque
from enum import Enum, auto
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgramState(Enum):
    WAITING_FOR_SURFACE = auto()
    DISPLAYING_WORD = auto()
    COUNTDOWN = auto()

@dataclass
class WordAttempt:
    target_word: str
    written_word: str
    similarity_score: int
    timestamp: float

class PartialMatch:
    def __init__(self, target_word: str):
        self.target_word = target_word
        self.current_text = ""
        self.last_update = time.time()
        
    def update(self, new_text: str) -> Tuple[bool, str, float]:
        """
        Updates the current text and returns match information
        Returns: (is_match, remaining_text, match_percentage)
        """
        self.current_text = new_text.upper()
        self.last_update = time.time()
        
        # Check if the current text is a prefix of the target word
        if self.target_word.startswith(self.current_text):
            match_percentage = len(self.current_text) / len(self.target_word)
            remaining = self.target_word[len(self.current_text):]
            return True, remaining, match_percentage
        
        # If not a prefix, calculate similarity
        similarity = SequenceMatcher(None, self.current_text, 
                                   self.target_word[:len(self.current_text)]).ratio()
        return False, self.target_word, similarity

class Config:
    """Configuration class to store all settings"""
    FRAME_SIZE = (640, 480)
    MODEL_INPUT_SIZE = (128, 128)
    SKIP_FRAMES = 3
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OCR_CONFIDENCE_THRESHOLD = 0.7
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.25
    COUNTDOWN_TIME = 5
    HISTORY_SIZE = 5
    OVERLAY_FONT_SCALE = 2.0
    OVERLAY_ALPHA = 0.6
    MATCH_TIMEOUT = 1.0  # seconds
    
    @classmethod
    def setup_torch(cls):
        torch.set_float32_matmul_precision('medium')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


@dataclass
class WordRecommendation:
    word: str
    difficulty_level: float  # 0-1 scale
    # category: str  # e.g., 'sight_word', 'phonetic', 'challenge'
    # reason: str

class WordDifficulty:
    """Analyzes word difficulty for dyslexic readers"""
    
    COMMON_CONFUSIONS = {
        'b': 'd', 'd': 'b', 'p': 'q', 'q': 'p',
        'm': 'w', 'w': 'm', 'n': 'u', 'u': 'n'
    }
    
    @staticmethod
    def calculate_difficulty(word: str) -> float:
        word = word.lower()
        score = 0.0
        
        # Length factor (longer words are harder)
        score += len(word) * 0.1
        
        # Commonly confused letters
        confused_letters = sum(1 for c in word if c in WordDifficulty.COMMON_CONFUSIONS)
        score += confused_letters * 0.2
        
        # Similar consecutive letters
        for i in range(len(word)-1):
            if word[i] == word[i+1]:
                score += 0.15
                
        # Normalize score to 0-1
        return min(score, 1.0)


class TinyLlamaGenerator:
    """Manages word generation using TinyLlama"""
    
    def __init__(self, model_path: str ="Qwen/Qwen1.5-0.5B-Chat-GGUF"):
        """
        Initialize the TinyLlama model
        model_path: Path to the llama.cpp compatible model file
        """
        try:
            self.llm = Llama.from_pretrained(
                repo_id=model_path,
                filename="*q8_0.gguf",
                tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
                    "Qwen/Qwen1.5-0.5B"
                ),
                # n_threads=4,
                verbose=False
            )
            logger.info("TinyLlama model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TinyLlama model: {e}")
            raise

    def generate_prompt(self, recent_scores: List[Dict], target_difficulty: float) -> str:
        """Create a prompt based on recent performance"""
        prompt = """You're dyslexia expert, can you please recommend one word the dyslexic person can practice writing to improve their condition for the next iteration. Their past writing experience is the following\n"""
        
        if recent_scores:
            for score in recent_scores:
                prompt += f"The dyslexic person has score of {score['score']} in word {score['word']}"
        else:
            prompt += "No previous attempts.\n"
        
        prompt += f"Please recommend words that a dyslexic person can struggle with and do not repeat same words twice"
        response_format = {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                },
                "required": ["word"],
            },
        }

        return (prompt, response_format) 

    def get_next_word(self, recent_scores: List[Dict], 
                      target_difficulty: float) -> Optional[WordRecommendation]:
        """Generate next word recommendation based on performance"""
        try:
            prompt = self.generate_prompt(recent_scores, target_difficulty)
            # Generate response with constrained parameters
            response = self.llm.create_chat_completion(
                messages = [{"role" :"user", "content":prompt[0]}],
                response_format = prompt[1],
                temperature=0.7,
                top_p=0.9,
                stream=True
            )
            
            json_res = ""
            # Get the generated word and clean it
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if "content" not in delta:
                    continue
                json_res += delta["content"]

            json_res = json.loads(json_res)

            generated_word = json_res['word']

            print(json_res)
            
            if generated_word:
                return WordRecommendation(
                    word=generated_word,
                    difficulty_level=target_difficulty
                )
            else:
                logger.warning("Generated word was empty or invalid")
                return None
            
        except Exception as e:
            logger.error(f"Word generation failed: {e}")
            return None


class AdaptiveWordSequence:
    """Manages word sequence with adaptive difficulty"""
    
    def __init__(self, model_path: str = "Qwen/Qwen1.5-0.5B-Chat-GGUF", history_size: int = 5):
        self.generator = TinyLlamaGenerator(model_path)
        self.history: deque = deque(maxlen=history_size)
        self.current_word: Optional[str] = None
        self.target_difficulty = 0.5  # Start at medium difficulty
        self.adjustment_rate = 0.1
        self._generate_next_word()  # Generate first word immediately
        self.partial_match = PartialMatch(self.get_current_word())
        
    def get_current_word(self) -> str:
        """Get current target word"""
        return self.current_word if self.current_word else "HELLO"

    def update_partial_match(self, text: str) -> Tuple[bool, str, float]:
        """Wrapper for partial match update"""
        return self.partial_match.update(text)

    def get_history(self) -> List[WordAttempt]:
        return list(self.history)
    
    def _generate_next_word(self):
        """Generate next word based on performance history"""
        recent_scores = [
            {"word": h.target_word, "score": h.similarity_score}
            for h in self.history
        ]
        
        # Adjust difficulty based on recent performance
        if recent_scores:
            avg_score = np.mean([s['score'] for s in recent_scores])
            if avg_score > 85:  # Increase difficulty if doing well
                self.target_difficulty = min(1.0, self.target_difficulty + self.adjustment_rate)
            elif avg_score < 65:  # Decrease difficulty if struggling
                self.target_difficulty = max(0.0, self.target_difficulty - self.adjustment_rate)
        
        recommendation = self.generator.get_next_word(recent_scores, self.target_difficulty)
        if recommendation:
            self.current_word = recommendation.word
            # Create new partial match for new word
            self.partial_match = PartialMatch(self.current_word)
            logger.info(f"Generated new word: {self.current_word} "
                       f"(Difficulty: {recommendation.difficulty_level:.2f})")
        else:
            # Fallback to default word if generation fails
            self.current_word = "HELLO"
            self.partial_match = PartialMatch(self.current_word)
            logger.warning("Using fallback word due to generation failure")
    
    def complete_current_word(self, written_word: str, similarity_score: int):
        """Record attempt and generate next word"""
        # First record the attempt
        self.history.append(WordAttempt(
            target_word=self.current_word,
            written_word=written_word,
            similarity_score=int(similarity_score * 100),  # Convert to percentage
            timestamp=time.time()
        ))
        # Then generate next word
        self._generate_next_word()

class GDINO:
    """Handles object detection using Grounding DINO model"""
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base"):
        self.model_id = model_id
        self.build_model()
        
    def build_model(self):
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16
            ).to(Config.DEVICE)
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise

    @torch.no_grad()
    @torch.autocast("cuda")
    def predict(self, pil_images: List[Image.Image], text_prompt: str) -> Optional[dict]:
        try:
            inputs = self.processor(
                images=pil_images, 
                text=text_prompt, 
                return_tensors="pt"
            ).to(Config.DEVICE)
            
            inputs = {key: (val.half() if val.dtype == torch.float32 else val) 
                     for key, val in inputs.items()}
            
            outputs = self.model(**inputs)
            return self.processor.post_process_grounded_object_detection(
                outputs,
                inputs['input_ids'],
                box_threshold=Config.BOX_THRESHOLD,
                text_threshold=Config.TEXT_THRESHOLD,
                target_sizes=[img.size[::-1] for img in pil_images]
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

class WritingAssistant:
    """Main class handling the writing assistance functionality"""
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.model = GDINO()
        self.word_sequence = AdaptiveWordSequence()
        self.state = ProgramState.WAITING_FOR_SURFACE
        self.countdown_start = None
        self.current_box = None
        
    def capture_frame(self, cap) -> Optional[np.ndarray]:
        ret, frame = cap.read()
        return cv2.resize(frame, Config.FRAME_SIZE) if ret else None

    def add_text_overlay(self, frame: np.ndarray, text: str, box: Tuple[int, int, int, int],
                        font_scale: float = 1.0, color: Tuple[int, int, int] = (192, 192, 192),
                        thickness: int = 2, alpha: float = 0.6) -> np.ndarray:
        """Add semi-transparent text overlay to the frame"""
        x1, y1, x2, y2 = box
        overlay = frame.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        
        cv2.putText(overlay, text, (text_x, text_y), font, font_scale,
                   color, thickness, cv2.LINE_AA)
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_resized = cv2.resize(frame, Config.MODEL_INPUT_SIZE)
        frame_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        results = self.model.predict([frame_pil], ["whiteboard. book"])
        
        if not results or not results[0]['boxes'].size(0):
            self.current_box = None
            self.state = ProgramState.WAITING_FOR_SURFACE
            return self.add_text_overlay(frame, "Waiting for writing surface...",
                                       (10, frame.shape[0]//2-30, frame.shape[1]-10, frame.shape[0]//2+30))
        
        return self.handle_detection(frame, results)

    def handle_detection(self, frame: np.ndarray, results: dict) -> np.ndarray:
        scale_x = frame.shape[1] / Config.MODEL_INPUT_SIZE[0]
        scale_y = frame.shape[0] / Config.MODEL_INPUT_SIZE[1]
        
        box = results[0]['boxes'][0]
        x1, y1, x2, y2 = map(int, [
            box[0] * scale_x, box[1] * scale_y,
            box[2] * scale_x, box[3] * scale_y
        ])
        self.current_box = (x1, y1, x2, y2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if self.state == ProgramState.COUNTDOWN:
            return self.handle_countdown_state(frame)
        elif self.state == ProgramState.DISPLAYING_WORD:
            return self.handle_writing_state(frame)
        else:
            self.state = ProgramState.DISPLAYING_WORD
            return frame

    def handle_countdown_state(self, frame: np.ndarray) -> np.ndarray:
        if self.countdown_start is None:
            self.countdown_start = time.time()
            
        elapsed = time.time() - self.countdown_start
        remaining = max(0, Config.COUNTDOWN_TIME - elapsed)
        
        if remaining == 0:
            self.state = ProgramState.DISPLAYING_WORD
            self.countdown_start = None
            return frame
            
        countdown_text = f"Next word in {int(remaining)}..."
        return self.add_text_overlay(frame, countdown_text, self.current_box,
                                   Config.OVERLAY_FONT_SCALE)

    def handle_writing_state(self, frame: np.ndarray) -> np.ndarray:
        if self.current_box is None:
            return frame
            
        x1, y1, x2, y2 = self.current_box
        cropped_content = frame[y1:y2, x1:x2]
        
        if cropped_content.size > 0:
            ocr_results = self.perform_ocr(cropped_content, x1, y1)
            frame = self.process_ocr_results(frame, ocr_results)
            
        # Add word overlay and history
        frame = self.add_text_overlay(frame, self.word_sequence.get_current_word(),
                                    self.current_box, Config.OVERLAY_FONT_SCALE)
        frame = self.display_history(frame)
        
        return frame

    def perform_ocr(self, cropped_image: np.ndarray, offset_x: int, offset_y: int) -> List:
        try:
            pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            result = self.ocr.ocr(np.array(pil_image), cls=True)
            
            if not result[0]:
                return []
                
            return [
                (
                    [(int(coord[0] + offset_x), int(coord[1] + offset_y)) 
                     for coord in coords],
                    (text, confidence)
                )
                for coords, (text, confidence) in result[0]
            ]
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []

    def process_ocr_results(self, frame: np.ndarray, ocr_results: List) -> np.ndarray:
        if self.state != ProgramState.DISPLAYING_WORD:
            return frame

        for coords, (text, confidence) in ocr_results:
            if confidence > Config.OCR_CONFIDENCE_THRESHOLD:
                detected_text = text.strip()
                
                # Update partial match
                is_match, remaining, match_percentage = self.word_sequence.update_partial_match(detected_text)
                if detected_text == target_word:
                    similarity_score = 1.0  # 100% for exact match
                else:
                    similarity_score = SequenceMatcher(None, detected_text, target_word).ratio()

                # Draw the detected text and match information
                color = (0, 255, 0) if is_match else (0, 0, 255)
                cv2.putText(frame, f"Detected: {detected_text}", (coords[0][0], coords[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # If exact match with target word, complete it and move to countdown
                target_word = self.word_sequence.get_current_word()
                if detected_text == target_word:
                    self.word_sequence.complete_current_word(detected_text, similarity_score)
                    self.state = ProgramState.COUNTDOWN
                    logger.info(f"Word completed: {detected_text}, Score: {match_percentage}")
                    break
                
                # Show remaining letters to write
                if is_match and remaining:
                    color = (192, 192, 192)  # Grey color for remaining letters
                    cv2.putText(frame, remaining, (coords[0][0], coords[0][1] + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    def display_history(self, frame: np.ndarray) -> np.ndarray:
        y_offset = 30
        for attempt in self.word_sequence.get_history():
            text = f"{attempt.target_word}: {attempt.written_word} ({attempt.similarity_score}%)"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            y_offset += 25
        return frame

def main():
    """Main application entry point"""
    try:
        Config.setup_torch()
        assistant = WritingAssistant()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
            
        frame_count = 0
        while True:
            frame = assistant.capture_frame(cap)
            if frame is None:
                logger.error("Failed to capture frame")
                break
                
            if frame_count % Config.SKIP_FRAMES == 0:
                frame = assistant.process_frame(frame)
                cv2.imshow("Writing Assistant", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
