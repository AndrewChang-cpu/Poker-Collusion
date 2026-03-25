import os                                                                                                                                                                         
import sys                                                                                                                                                                        
import time                                                         
                                        
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)                                                                                                                                                
                                                                    
from poker_collusion.cfr import CFRTrainer
from poker_collusion.env import (
    deal_new_hand, get_current_player, get_legal_actions,                                                                                                                         
    get_info_key, is_terminal, get_payoffs, apply_action,
    is_chance_node, sample_chance,                                                                                                                                                
)                                                                   
                                                                                                                                                                                
class GameModule:
    deal_new_hand = staticmethod(deal_new_hand)                                                                                                                                   
    get_current_player = staticmethod(get_current_player)           
    get_legal_actions = staticmethod(get_legal_actions)                                                                                                                           
    get_info_key = staticmethod(get_info_key)
    is_terminal = staticmethod(is_terminal)                                                                                                                                       
    get_payoffs = staticmethod(get_payoffs)                         
    apply_action = staticmethod(apply_action)                                                                                                                                     
    is_chance_node = staticmethod(is_chance_node)
    sample_chance = staticmethod(sample_chance)                                                                                                                                   
                                                                    
ITERS = 4                                                                                                                                                                        
BATCH = 24  # multiple of 3
                                                                                                                                                                                
results = {}                                                        
for workers in [3, 2, 4, 8]:                                                                                                                                                      
    game = GameModule()                                             
    trainer = CFRTrainer(game, num_players=3)
    t0 = time.perf_counter()                                                                                                                                                      
    trainer.train_parallel(num_iterations=ITERS, num_workers=workers, batch_size=BATCH, log_interval=1)
    elapsed = time.perf_counter() - t0                                                                                                                                            
    tps = (ITERS * BATCH) / elapsed                                 
    results[workers] = tps                                                                                                                                                        
    print(f"workers={workers:2d} | {elapsed:.2f}s | {tps:.1f} traversals/s")
                                                                                                                                                                                
baseline = results[1]
print("\nSpeedup vs 1 worker:")                                                                                                                                                   
for w, tps in results.items():                                      
    print(f"  {w} workers: {tps/baseline:.2f}x  (ideal: {w:.2f}x)")