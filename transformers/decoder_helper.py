import torch 
import numpy as np

def mi(logits_teacher, logits_student, **kwargs):
    '''
    Adjust logits post-hoc. 
    '''
    logp_teacher = torch.log_softmax(logits_teacher,dim=-1)
    logp_student = torch.log_softmax(logits_student,dim=-1)
    p_teacher = logp_teacher.exp() 
    diff_num = (p_teacher - logp_student.exp())
    diff_demon = (logp_teacher - logp_student)
    result = (diff_demon * p_teacher / diff_num).log()
    print(result.shape) 
    return result

def post_process_easy(teacher_distribution, student_distribution, model_kwargs):
    next_token_scores = teacher_distribution - model_kwargs['student_coef'] * student_distribution
    return next_token_scores 


def post_process_reweight(  input_ids, next_indices, next_tokens, next_token_scores, student_scores, 
                            model_kwargs, main_model):
    # new_input_ids = input_ids[next_indices.squeeze(0)]
    # temp_rollout = torch.cat([new_input_ids, next_tokens.view(new_input_ids.size(0), -1)], dim=-1)
    # preseqlen = temp_rollout.size(1)

    # score_model = model_kwargs["score_model"]

    # lookahead_steps = score_model.lookahead
    # pad_token_id = main_model.config.pad_token_id
    # eos_token_id = main_model.config.eos_token_id

    # log diff. 
    # logreweight = next_token_scores - student_scores 

    # diff = torch.abs(next_token_scores - student_scores)
    # print(diff) 

    # print(next_token_scores.shape, student_scores.shape, )
    teacher_scores = next_token_scores
    max_log_prob = torch.max(teacher_scores, dim=-1).values.reshape(-1,1)
    max_prob = max_log_prob.exp()

    if model_kwargs['student_alpha'] == 0:
        thres = -np.inf + max_log_prob
    else:
        thres = np.log(model_kwargs['student_alpha']) + max_log_prob

    # calculating dynamic weight
    if model_kwargs['use_dynamic_coef']:
        student_coef = 1- torch.pow(max_prob, model_kwargs['student_coef'])
    else:
        student_coef = model_kwargs['student_coef']

    # thresholding
    prob_cond = teacher_scores >= thres
    next_token_scores = torch.where(prob_cond, next_token_scores, next_token_scores - 20)
    trunc_cond = teacher_scores < thres
    next_token_scores = torch.where(trunc_cond, next_token_scores, next_token_scores - student_coef * student_scores)

    # print(next_token_scores) # analysis 



    next_token_scores, next_tokens_reorder = torch.topk(
        next_token_scores, next_token_scores.size(1), dim=-1, largest=True
    )
    # print(next_tokens_reorder) # analysis 
    next_tokens = torch.index_select(next_tokens, 1, next_tokens_reorder.view(-1))
    next_indices = torch.index_select(next_indices, 1, next_tokens_reorder.view(-1))
    return next_indices, next_tokens, next_token_scores


def post_process_reweight_v2(  input_ids, next_indices, next_tokens, next_token_scores, student_scores, 
                            model_kwargs, main_model, teacher_scores):
    # print('post_process_reweight_v2')
    
    teacher_token_scores_exp = teacher_scores.exp()
    student_scores_exp = student_scores.exp()
    alpha = 1.0 
    # print(student_scores_exp, teacher_token_scores_exp)
    posterior_distrib = teacher_token_scores_exp / (student_scores_exp  + alpha * teacher_token_scores_exp)
    # print(posterior_distrib, next_token_scores, torch.min(next_token_scores)) 
    diff = teacher_scores - student_scores 
    # next_token_scores = torch.where(posterior_distrib < 0.5, next_token_scores, torch.tensor(float('-inf')).to(next_token_scores.device)) 
    next_token_scores = torch.where(posterior_distrib < 0.3, next_token_scores, next_token_scores - 20 - teacher_scores - torch.max(diff) + diff ) 

    next_token_scores, next_tokens_reorder = torch.topk(
        next_token_scores, next_token_scores.size(1), dim=-1, largest=True
    )
    # print(next_tokens_reorder) # analysis 
    next_tokens = torch.index_select(next_tokens, 1, next_tokens_reorder.view(-1))
    next_indices = torch.index_select(next_indices, 1, next_tokens_reorder.view(-1))
    return next_indices, next_tokens, next_token_scores