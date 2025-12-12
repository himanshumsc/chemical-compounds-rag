#!/usr/bin/env python3
"""
ResponseCombiner: Handles intelligent merging and formatting of responses from both models
Part of the modular multimodal RAG system
"""

import json
import logging
from typing import Dict, Any, List, Optional
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseCombiner:
    """Combines and formats responses from both Phi-4 and Qwen models"""
    
    def __init__(self):
        self.combination_strategies = {
            'side_by_side': self._format_side_by_side,
            'intelligent_merge': self._intelligent_merge,
            'json_format': self._format_json,
            'detailed_comparison': self._format_detailed_comparison
        }
        
        logger.info("ResponseCombiner initialized")
    
    def combine_responses(self, generation_result: Dict[str, Any], 
                         query: str, strategy: str = 'side_by_side') -> Dict[str, Any]:
        """
        Combine responses from both models using specified strategy
        
        Args:
            generation_result: Result from ParallelGenerator
            query: Original user query
            strategy: Combination strategy ('side_by_side', 'intelligent_merge', 'json_format', 'detailed_comparison')
        
        Returns:
            Combined and formatted response
        """
        logger.info(f"Combining responses using strategy: {strategy}")
        
        # Extract individual results
        phi4_result = generation_result.get('phi4_result')
        qwen_result = generation_result.get('qwen_result')
        
        # Check if we have any successful results
        phi4_success = phi4_result and phi4_result.get('success', False)
        qwen_success = qwen_result and qwen_result.get('success', False)
        
        if not phi4_success and not qwen_success:
            return self._format_error_response(generation_result)
        
        # Apply combination strategy
        if strategy in self.combination_strategies:
            formatted_response = self.combination_strategies[strategy](
                phi4_result, qwen_result, query, generation_result
            )
        else:
            logger.warning(f"Unknown strategy {strategy}, using side_by_side")
            formatted_response = self.combination_strategies['side_by_side'](
                phi4_result, qwen_result, query, generation_result
            )
        
        # Add metadata
        formatted_response['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'strategy_used': strategy,
            'phi4_success': phi4_success,
            'qwen_success': qwen_success,
            'total_generation_time': generation_result.get('total_time', 0),
            'models_used': generation_result.get('models_used', [])
        }
        
        return formatted_response
    
    def _format_side_by_side(self, phi4_result: Optional[Dict], qwen_result: Optional[Dict], 
                            query: str, generation_result: Dict) -> Dict[str, Any]:
        """Format responses side by side with clear separation"""
        
        formatted_text = f"=== QUERY: {query} ===\n\n"
        
        # Phi-4 Response Section
        if phi4_result and phi4_result.get('success'):
            formatted_text += "🤖 PHI-4 RESPONSE:\n"
            formatted_text += "=" * 50 + "\n"
            formatted_text += phi4_result['response'] + "\n\n"
            formatted_text += f"⏱️ Generation Time: {phi4_result.get('generation_time', 0):.2f}s\n"
            formatted_text += f"🖼️ Images Processed: {phi4_result.get('images_processed', 0)}\n\n"
        elif phi4_result:
            formatted_text += "❌ PHI-4 RESPONSE (FAILED):\n"
            formatted_text += "=" * 50 + "\n"
            formatted_text += f"Error: {phi4_result.get('error', 'Unknown error')}\n\n"
        
        # Qwen Response Section
        if qwen_result and qwen_result.get('success'):
            formatted_text += "🤖 QWEN RESPONSE:\n"
            formatted_text += "=" * 50 + "\n"
            formatted_text += qwen_result['response'] + "\n\n"
            formatted_text += f"⏱️ Generation Time: {qwen_result.get('generation_time', 0):.2f}s\n"
            formatted_text += f"🖼️ Images Processed: {qwen_result.get('images_processed', 0)}\n\n"
        elif qwen_result:
            formatted_text += "❌ QWEN RESPONSE (FAILED):\n"
            formatted_text += "=" * 50 + "\n"
            formatted_text += f"Error: {qwen_result.get('error', 'Unknown error')}\n\n"
        
        # Summary Section
        formatted_text += "📊 SUMMARY:\n"
        formatted_text += "=" * 50 + "\n"
        formatted_text += f"Total Processing Time: {generation_result.get('total_time', 0):.2f}s\n"
        formatted_text += f"Models Used: {', '.join(generation_result.get('models_used', []))}\n"
        
        return {
            'formatted_text': formatted_text,
            'strategy': 'side_by_side',
            'phi4_response': phi4_result.get('response') if phi4_result else None,
            'qwen_response': qwen_result.get('response') if qwen_result else None
        }
    
    def _intelligent_merge(self, phi4_result: Optional[Dict], qwen_result: Optional[Dict], 
                          query: str, generation_result: Dict) -> Dict[str, Any]:
        """Intelligently merge responses from both models"""
        
        # Extract successful responses
        phi4_response = phi4_result.get('response') if phi4_result and phi4_result.get('success') else None
        qwen_response = qwen_result.get('response') if qwen_result and qwen_result.get('success') else None
        
        if not phi4_response and not qwen_response:
            return self._format_error_response(generation_result)
        
        # If only one response, return it
        if not phi4_response:
            return self._format_single_response('Qwen', qwen_response, qwen_result, generation_result)
        if not qwen_response:
            return self._format_single_response('Phi-4', phi4_response, phi4_result, generation_result)
        
        # Both responses available - intelligent merging
        merged_response = self._merge_responses_intelligently(phi4_response, qwen_response, query)
        
        formatted_text = f"=== QUERY: {query} ===\n\n"
        formatted_text += "🧠 INTELLIGENTLY MERGED RESPONSE:\n"
        formatted_text += "=" * 50 + "\n"
        formatted_text += merged_response + "\n\n"
        
        formatted_text += "📋 SOURCE BREAKDOWN:\n"
        formatted_text += "=" * 50 + "\n"
        formatted_text += "🤖 Phi-4 Contribution:\n"
        formatted_text += phi4_response[:200] + "..." if len(phi4_response) > 200 else phi4_response
        formatted_text += "\n\n🤖 Qwen Contribution:\n"
        formatted_text += qwen_response[:200] + "..." if len(qwen_response) > 200 else qwen_response
        formatted_text += "\n\n"
        
        formatted_text += f"⏱️ Total Processing Time: {generation_result.get('total_time', 0):.2f}s\n"
        
        return {
            'formatted_text': formatted_text,
            'strategy': 'intelligent_merge',
            'merged_response': merged_response,
            'phi4_response': phi4_response,
            'qwen_response': qwen_response
        }
    
    def _merge_responses_intelligently(self, phi4_response: str, qwen_response: str, query: str) -> str:
        """Intelligently merge two responses"""
        
        # Extract key information from both responses
        phi4_key_points = self._extract_key_points(phi4_response)
        qwen_key_points = self._extract_key_points(qwen_response)
        
        # Combine unique key points
        all_key_points = list(set(phi4_key_points + qwen_key_points))
        
        # Create merged response
        merged = f"Based on analysis from both AI models, here's a comprehensive answer:\n\n"
        
        # Add key points
        if all_key_points:
            merged += "🔑 Key Points:\n"
            for i, point in enumerate(all_key_points[:5], 1):  # Limit to top 5
                merged += f"{i}. {point}\n"
            merged += "\n"
        
        # Add detailed information (prefer longer, more detailed response)
        if len(phi4_response) > len(qwen_response):
            merged += "📖 Detailed Information:\n"
            merged += phi4_response
        else:
            merged += "📖 Detailed Information:\n"
            merged += qwen_response
        
        return merged
    
    def _extract_key_points(self, response: str) -> List[str]:
        """Extract key points from a response"""
        # Simple extraction based on common patterns
        key_points = []
        
        # Look for bullet points
        bullet_pattern = r'[•\-\*]\s*([^\n]+)'
        bullets = re.findall(bullet_pattern, response)
        key_points.extend(bullets)
        
        # Look for numbered lists
        numbered_pattern = r'\d+\.\s*([^\n]+)'
        numbered = re.findall(numbered_pattern, response)
        key_points.extend(numbered)
        
        # Look for sentences with key terms
        sentences = response.split('.')
        for sentence in sentences:
            if any(term in sentence.lower() for term in ['important', 'key', 'main', 'primary', 'essential']):
                key_points.append(sentence.strip())
        
        return key_points[:10]  # Limit to 10 key points
    
    def _format_single_response(self, model_name: str, response: str, result: Dict, generation_result: Dict) -> Dict[str, Any]:
        """Format response from a single model"""
        formatted_text = f"=== QUERY: {generation_result.get('query', 'Unknown')} ===\n\n"
        formatted_text += f"🤖 {model_name.upper()} RESPONSE:\n"
        formatted_text += "=" * 50 + "\n"
        formatted_text += response + "\n\n"
        formatted_text += f"⏱️ Generation Time: {result.get('generation_time', 0):.2f}s\n"
        formatted_text += f"🖼️ Images Processed: {result.get('images_processed', 0)}\n"
        
        return {
            'formatted_text': formatted_text,
            'strategy': 'single_model',
            'model_used': model_name,
            'response': response
        }
    
    def _format_json(self, phi4_result: Optional[Dict], qwen_result: Optional[Dict], 
                    query: str, generation_result: Dict) -> Dict[str, Any]:
        """Format responses as structured JSON"""
        
        json_response = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'generation_stats': {
                'total_time': generation_result.get('total_time', 0),
                'models_used': generation_result.get('models_used', [])
            },
            'responses': {}
        }
        
        if phi4_result:
            json_response['responses']['phi4'] = {
                'success': phi4_result.get('success', False),
                'response': phi4_result.get('response'),
                'generation_time': phi4_result.get('generation_time', 0),
                'images_processed': phi4_result.get('images_processed', 0),
                'error': phi4_result.get('error')
            }
        
        if qwen_result:
            json_response['responses']['qwen'] = {
                'success': qwen_result.get('success', False),
                'response': qwen_result.get('response'),
                'generation_time': qwen_result.get('generation_time', 0),
                'images_processed': qwen_result.get('images_processed', 0),
                'error': qwen_result.get('error')
            }
        
        return {
            'formatted_text': json.dumps(json_response, indent=2),
            'strategy': 'json_format',
            'json_data': json_response
        }
    
    def _format_detailed_comparison(self, phi4_result: Optional[Dict], qwen_result: Optional[Dict], 
                                  query: str, generation_result: Dict) -> Dict[str, Any]:
        """Format detailed comparison of both responses"""
        
        formatted_text = f"=== DETAILED COMPARISON FOR: {query} ===\n\n"
        
        # Performance comparison
        formatted_text += "📊 PERFORMANCE COMPARISON:\n"
        formatted_text += "=" * 50 + "\n"
        
        if phi4_result:
            formatted_text += f"🤖 Phi-4: {phi4_result.get('generation_time', 0):.2f}s"
            formatted_text += f" | Success: {phi4_result.get('success', False)}"
            formatted_text += f" | Images: {phi4_result.get('images_processed', 0)}\n"
        
        if qwen_result:
            formatted_text += f"🤖 Qwen: {qwen_result.get('generation_time', 0):.2f}s"
            formatted_text += f" | Success: {qwen_result.get('success', False)}"
            formatted_text += f" | Images: {qwen_result.get('images_processed', 0)}\n"
        
        formatted_text += f"⏱️ Total Time: {generation_result.get('total_time', 0):.2f}s\n\n"
        
        # Response comparison
        formatted_text += "📝 RESPONSE COMPARISON:\n"
        formatted_text += "=" * 50 + "\n"
        
        if phi4_result and phi4_result.get('success'):
            formatted_text += "🤖 PHI-4 RESPONSE:\n"
            formatted_text += "-" * 30 + "\n"
            formatted_text += phi4_result['response'] + "\n\n"
        
        if qwen_result and qwen_result.get('success'):
            formatted_text += "🤖 QWEN RESPONSE:\n"
            formatted_text += "-" * 30 + "\n"
            formatted_text += qwen_result['response'] + "\n\n"
        
        # Analysis
        formatted_text += "🔍 ANALYSIS:\n"
        formatted_text += "=" * 50 + "\n"
        
        if phi4_result and qwen_result and phi4_result.get('success') and qwen_result.get('success'):
            phi4_len = len(phi4_result['response'])
            qwen_len = len(qwen_result['response'])
            
            formatted_text += f"• Phi-4 response length: {phi4_len} characters\n"
            formatted_text += f"• Qwen response length: {qwen_len} characters\n"
            formatted_text += f"• More detailed: {'Phi-4' if phi4_len > qwen_len else 'Qwen'}\n"
            formatted_text += f"• Faster generation: {'Phi-4' if phi4_result.get('generation_time', 0) < qwen_result.get('generation_time', 0) else 'Qwen'}\n"
        
        return {
            'formatted_text': formatted_text,
            'strategy': 'detailed_comparison',
            'phi4_response': phi4_result.get('response') if phi4_result else None,
            'qwen_response': qwen_result.get('response') if qwen_result else None
        }
    
    def _format_error_response(self, generation_result: Dict) -> Dict[str, Any]:
        """Format error response when both models fail"""
        formatted_text = "❌ ERROR: Both models failed to generate responses\n\n"
        formatted_text += "🔍 DEBUGGING INFORMATION:\n"
        formatted_text += "=" * 50 + "\n"
        
        phi4_result = generation_result.get('phi4_result')
        qwen_result = generation_result.get('qwen_result')
        
        if phi4_result:
            formatted_text += f"🤖 Phi-4 Error: {phi4_result.get('error', 'Unknown error')}\n"
        
        if qwen_result:
            formatted_text += f"🤖 Qwen Error: {qwen_result.get('error', 'Unknown error')}\n"
        
        formatted_text += f"⏱️ Total Time: {generation_result.get('total_time', 0):.2f}s\n"
        
        return {
            'formatted_text': formatted_text,
            'strategy': 'error',
            'error': True
        }
    
    def convert_json_to_formatted_text(self, json_data: Dict[str, Any]) -> str:
        """Convert JSON response to formatted text"""
        if 'formatted_text' in json_data:
            return json_data['formatted_text']
        
        # If it's raw JSON data, format it
        formatted_text = "=== JSON RESPONSE ===\n\n"
        formatted_text += json.dumps(json_data, indent=2)
        
        return formatted_text
