#######

# Need to update total scoring method

########



import os
import numpy as np
import pandas as pd


   
def appendInfo(path, filename, index, validity):
    global out
    global origin_x, origin_y
    global positive_end_x, positive_end_y
    global active_end_x, active_end_y
    global negative_end_x, negative_end_y
    global passive_end_x, passive_end_y
    global empty_vid_x, empty_vid_y
    global empty_aud_x, empty_aud_y
    global empty_tex_x, empty_tex_y
    global neutral_target_start_x, neutral_target_start_y
    global neutral_target_end_x, neutral_target_end_y
    global total_videos
    global w_vid, w_text, w_aud
    global s_posAct, s_posInact, s_negInact, s_negAct
    suffix = ''
    if validity == '2':
        suffix = ''
        correct_suffix = '?'
    else:
        suffix = '_' + validity
        correct_suffix = '_' + str(validity) + '?'
    
    title_suffix = ' (' + validity + ' s)'
    
    mismatch_suffix = '_' + validity

    mismatch_str = 'Mismatch' + mismatch_suffix
    
    
    print("Calculating results for " + f"{index+1}" + "/" + f"{total_videos}: " + filename.replace("_processed_all_data.csv","") + " with " + validity + " seconds validity")
    #print(out.shape)
    df = pd.read_csv(path + "//" + filename)
    
    #print(df)
    
    n_frames = int(df.iloc[-1,0])
    out.loc[index, "Video filename"] = str(filename.replace("_processed_all_data.csv",""))
    out.loc[index, "Total frames"] = int(n_frames)

    n_neutral_frames = 0
    
    n_nonNeutral_frames = 0
    
    n_frames_with_face = 0
    
    n_sound_frames = 0
    
    n_sound_frames_with_face = 0
    
    n_possible_mismatch_frames = 0
    
    n_scorable_frames = 0
    
    n_pos_frames = 0
    n_neg_frames = 0

    n_act_frames = 0
    n_inact_frames = 0
    
    n_posAct_frames = 0
    n_negAct_frames = 0

    n_posInact_frames = 0
    n_negInact_frames = 0
    
    
    
    n_pos_frames_wFace = 0
    n_neg_frames_wFace = 0

    n_act_frames_wFace = 0
    n_inact_frames_wFace = 0
    
    n_posAct_frames_wFace = 0
    n_negAct_frames_wFace = 0

    n_posInact_frames_wFace = 0
    n_negInact_frames_wFace = 0
    
    
    
    n_pos_frames_from_vid = 0
    n_neg_frames_from_vid = 0

    n_act_frames_from_vid = 0
    n_inact_frames_from_vid = 0
    
    n_posAct_frames_from_vid = 0
    n_negAct_frames_from_vid = 0

    n_posInact_frames_from_vid = 0
    n_negInact_frames_from_vid = 0
    
    n_pos_sound_frames = 0
    n_neg_sound_frames = 0
    
    n_act_sound_frames = 0
    n_inact_sound_frames = 0
    
    n_posAct_sound_frames = 0
    n_negAct_sound_frames = 0
    
    n_posInact_sound_frames = 0
    n_negInact_sound_frames = 0
    
    n_pos_sound_frames_from_aud = 0
    n_neg_sound_frames_from_aud = 0
    
    n_act_sound_frames_from_aud = 0
    n_inact_sound_frames_from_aud = 0
    
    n_posAct_sound_frames_from_aud = 0
    n_negAct_sound_frames_from_aud = 0
    
    n_posInact_sound_frames_from_aud = 0
    n_negInact_sound_frames_from_aud = 0
    
    n_pos_sound_frames_from_text = 0
    n_neg_sound_frames_from_text = 0
    
    n_act_sound_frames_from_text = 0
    n_inact_sound_frames_from_text = 0
    
    n_posAct_sound_frames_from_text = 0
    n_negAct_sound_frames_from_text = 0
    
    n_posInact_sound_frames_from_text = 0
    n_negInact_sound_frames_from_text = 0
    
    n_vid_corr_pos_frames = 0
    n_vid_corr_neg_frames = 0
    n_vid_corr_act_frames = 0
    n_vid_corr_inact_frames = 0
    
    n_vid_corr_posAct_frames = 0
    n_vid_corr_negAct_frames = 0
    n_vid_corr_posInact_frames = 0
    n_vid_corr_negInact_frames = 0
    
    n_vid_corr_tot_frames = 0
    
    n_aud_corr_pos_frames = 0
    n_aud_corr_neg_frames = 0
    n_aud_corr_act_frames = 0
    n_aud_corr_inact_frames = 0
    
    n_aud_corr_posAct_frames = 0
    n_aud_corr_negAct_frames = 0
    n_aud_corr_posInact_frames = 0
    n_aud_corr_negInact_frames = 0
    
    n_aud_corr_tot_frames = 0
    
    n_text_corr_pos_frames = 0
    n_text_corr_neg_frames = 0
    n_text_corr_act_frames = 0
    n_text_corr_inact_frames = 0
    
    n_text_corr_posAct_frames = 0
    n_text_corr_negAct_frames = 0
    n_text_corr_posInact_frames = 0
    n_text_corr_negInact_frames = 0
    
    n_text_corr_tot_frames = 0
    
    n_VidAudMismatch_frames = 0
    n_TextAudMismatch_frames = 0
    n_VidTextMismatch_frames = 0
    
    vid_score_tot = 0
    aud_score_tot = 0
    text_score_tot = 0
    score_tot = 0
    
    n_vid_cat2_bef_cat1 = 0
    n_vid_cat3_bef_cat1 = 0
    n_vid_cat4_bef_cat1 = 0
    
    n_vid_cat1_bef_cat2 = 0
    n_vid_cat3_bef_cat2 = 0
    n_vid_cat4_bef_cat2 = 0
    
    n_vid_cat2_bef_cat3 = 0
    n_vid_cat1_bef_cat3 = 0
    n_vid_cat4_bef_cat3 = 0
    
    n_vid_cat2_bef_cat4 = 0
    n_vid_cat3_bef_cat4 = 0
    n_vid_cat1_bef_cat4 = 0
    
    
    
    
    n_aud_cat2_bef_cat1 = 0
    n_aud_cat3_bef_cat1 = 0
    n_aud_cat4_bef_cat1 = 0
    
    n_aud_cat1_bef_cat2 = 0
    n_aud_cat3_bef_cat2 = 0
    n_aud_cat4_bef_cat2 = 0
    
    n_aud_cat2_bef_cat3 = 0
    n_aud_cat1_bef_cat3 = 0
    n_aud_cat4_bef_cat3 = 0
    
    n_aud_cat2_bef_cat4 = 0
    n_aud_cat3_bef_cat4 = 0
    n_aud_cat1_bef_cat4 = 0
    
    
    
    
    n_text_cat2_bef_cat1 = 0
    n_text_cat3_bef_cat1 = 0
    n_text_cat4_bef_cat1 = 0
    
    n_text_cat1_bef_cat2 = 0
    n_text_cat3_bef_cat2 = 0
    n_text_cat4_bef_cat2 = 0
    
    n_text_cat2_bef_cat3 = 0
    n_text_cat1_bef_cat3 = 0
    n_text_cat4_bef_cat3 = 0
    
    n_text_cat2_bef_cat4 = 0
    n_text_cat3_bef_cat4 = 0
    n_text_cat1_bef_cat4 = 0
    
    for j in range(0,df.shape[0]):
        
        
        this_score_vid = 0
        this_score_aud = 0
        this_score_text = 0
        
        aud_x_str = 'Aud_x' + suffix
        aud_y_str = 'Aud_y' + suffix
        text_x_str = 'Text_x' + suffix
        text_y_str = 'Text_y' + suffix
        
        aud_pred_str = 'Audio class' + suffix
        text_pred_str = 'Text class' + suffix
        video_pred_str = 'Video class'
        
        # aud_corr_str = 'Audio correct' + correct_suffix
        # text_corr_str = 'Text correct' + correct_suffix
        # video_corr_str = 'Video correct?'
        
        
        #### For text and audio
        
        if df.loc[j,text_pred_str] != 0:
            n_sound_frames += 1
            
            # if df.loc[j,aud_corr_str] == 1: 
                # n_aud_corr_tot_frames += 1
                
            # if df.loc[j,text_corr_str] == 1: 
                # n_text_corr_tot_frames += 1
                
                
                
                
            
            if df.loc[j,aud_pred_str] == 2 or df.loc[j,aud_pred_str] == 3:
                n_pos_sound_frames_from_aud += 1
            elif df.loc[j,aud_pred_str] == 1 or df.loc[j,aud_pred_str] == 4:
                n_neg_sound_frames_from_aud += 1
                
            if df.loc[j,aud_pred_str] == 3 or df.loc[j,aud_pred_str] == 4:
                n_act_sound_frames_from_aud += 1
            elif df.loc[j,aud_pred_str] == 1 or df.loc[j,aud_pred_str] == 2:
                n_inact_sound_frames_from_aud += 1
            


            
            if df.loc[j,text_pred_str] == 2 or df.loc[j,text_pred_str] == 3:
                n_pos_sound_frames_from_text += 1     
            elif df.loc[j,text_pred_str] == 1 or df.loc[j,text_pred_str] == 4:
                n_neg_sound_frames_from_text += 1  
                
            if df.loc[j,text_pred_str] == 3 or df.loc[j,text_pred_str] == 4:
                n_act_sound_frames_from_text += 1   
            elif df.loc[j,text_pred_str] == 1 or df.loc[j,text_pred_str] == 2:
                n_inact_sound_frames_from_text += 1
                
            
            
            
            
            
            
            if df.loc[j,aud_pred_str] == 1:
                n_negInact_sound_frames_from_aud += 1
                this_score_aud = s_negInact
            
            elif df.loc[j,aud_pred_str] == 2:
                n_posInact_sound_frames_from_aud += 1
                this_score_aud = s_posInact
            
            elif df.loc[j,aud_pred_str] == 3:
                n_posAct_sound_frames_from_aud += 1
                this_score_aud = s_posAct
            
            elif df.loc[j,aud_pred_str] == 4:
                n_negAct_sound_frames_from_aud += 1
                this_score_aud = s_negAct
                
                
                
            if df.loc[j,text_pred_str] == 1:
                n_negInact_sound_frames_from_text += 1
                this_score_text = s_negInact
            
            elif df.loc[j,text_pred_str] == 2:
                n_posInact_sound_frames_from_text += 1
                this_score_text = s_posInact
            
            elif df.loc[j,text_pred_str] == 3:
                n_posAct_sound_frames_from_text += 1
                this_score_text = s_posAct
            
            elif df.loc[j,text_pred_str] == 4:
                n_negAct_sound_frames_from_text += 1
                this_score_text = s_negAct
                
                
                
                
                
            # if df.loc[j,'Target class'] == 2 or df.loc[j,'Target class'] == 3:
                # n_pos_sound_frames += 1
                
                # if df.loc[j,aud_pred_str] == 2 or df.loc[j,aud_pred_str] == 3:
                    # n_aud_corr_pos_frames += 1
                    
                # if df.loc[j,text_pred_str] == 2 or df.loc[j,text_pred_str] == 3:
                    # n_text_corr_pos_frames += 1
                
            # elif df.loc[j,'Target class'] == 1 or df.loc[j,'Target class'] == 4:
                # n_neg_sound_frames += 1
                
                # if df.loc[j,aud_pred_str] == 1 or df.loc[j,aud_pred_str] == 4:
                    # n_aud_corr_neg_frames += 1
                    
                # if df.loc[j,text_pred_str] == 1 or df.loc[j,text_pred_str] == 4:
                    # n_text_corr_neg_frames += 1
                
            # if df.loc[j,'Target class'] == 3 or df.loc[j,'Target class'] == 4:
                # n_act_sound_frames += 1
                
                # if df.loc[j,aud_pred_str] == 3 or df.loc[j,aud_pred_str] == 4:
                    # n_aud_corr_act_frames += 1
                    
                # if df.loc[j,text_pred_str] == 3 or df.loc[j,text_pred_str] == 4:
                    # n_text_corr_act_frames += 1
                
            # elif df.loc[j,'Target class'] == 1 or df.loc[j,'Target class'] == 2:
                # n_inact_sound_frames += 1
                
                # if df.loc[j,aud_pred_str] == 1 or df.loc[j,aud_pred_str] == 2:
                    # n_aud_corr_inact_frames += 1
                    
                # if df.loc[j,text_pred_str] == 1 or df.loc[j,text_pred_str] == 2:
                    # n_text_corr_inact_frames += 1
                
                
                
                
            # if df.loc[j,'Target class'] == 1:
                # n_negInact_sound_frames += 1
                
                # if df.loc[j,aud_pred_str] == 1:
                    # n_aud_corr_negInact_frames += 1
                    
                # if df.loc[j,text_pred_str] == 1:
                    # n_text_corr_negInact_frames += 1
            
            # elif df.loc[j,'Target class'] == 2:
                # n_posInact_sound_frames += 1
                
                # if df.loc[j,aud_pred_str] == 2:
                    # n_aud_corr_posInact_frames += 1
                    
                # if df.loc[j,text_pred_str] == 2:
                    # n_text_corr_posInact_frames += 1
            
            # elif df.loc[j,'Target class'] == 3:
                # n_posAct_sound_frames += 1
                
                # if df.loc[j,aud_pred_str] == 3:
                    # n_aud_corr_posAct_frames += 1
                    
                # if df.loc[j,text_pred_str] == 3:
                    # n_text_corr_posAct_frames += 1
            
            # elif df.loc[j,'Target class'] == 4:
                # n_negAct_sound_frames += 1
                
                # if df.loc[j,aud_pred_str] == 4:
                    # n_aud_corr_negAct_frames += 1
                    
                # if df.loc[j,text_pred_str] == 4:
                    # n_text_corr_negAct_frames += 1
                    
            
            
            # if df.loc[j,'Target class'] == df.loc[j,aud_pred_str]: 
                # n_aud_corr_tot_frames += 1
                
            # if df.loc[j,'Target class'] == df.loc[j,text_pred_str]: 
                # n_text_corr_tot_frames += 1

        
        #### For video
        if df.loc[j,'Has face?'] == 1:
            n_frames_with_face += 1
            
            # if df.loc[j,video_corr_str] == 1: 
                # n_vid_corr_tot_frames += 1
            
            
            
            if df.loc[j,video_pred_str] == 2 or df.loc[j,video_pred_str] == 3:
                n_pos_frames_from_vid += 1
                
            elif df.loc[j,video_pred_str] == 1 or df.loc[j,video_pred_str] == 4:
                n_neg_frames_from_vid += 1
                
            if df.loc[j,video_pred_str] == 3 or df.loc[j,video_pred_str] == 4:
                n_act_frames_from_vid += 1
                
            elif df.loc[j,video_pred_str] == 1 or df.loc[j,video_pred_str] == 2:
                n_inact_frames_from_vid += 1
                

            if df.loc[j,video_pred_str] == 1:
                n_negInact_frames_from_vid += 1
                this_score_vid = s_negInact
            
            elif df.loc[j,video_pred_str] == 2:
                n_posInact_frames_from_vid += 1
                this_score_vid = s_posInact
            
            elif df.loc[j,video_pred_str] == 3:
                n_posAct_frames_from_vid += 1
                this_score_vid = s_posAct
            
            elif df.loc[j,video_pred_str] == 4:
                n_negAct_frames_from_vid += 1
                this_score_vid = s_negAct
                
            



            
            # if df.loc[j,'Target class'] == 2 or df.loc[j,'Target class'] == 3:

                
                # if df.loc[j,video_pred_str] == 2 or df.loc[j,video_pred_str] == 3:
                    # n_vid_corr_pos_frames += 1
                
            # elif df.loc[j,'Target class'] == 1 or df.loc[j,'Target class'] == 4:

                
                # if df.loc[j,video_pred_str] == 1 or df.loc[j,video_pred_str] == 4:
                    # n_vid_corr_neg_frames += 1
                
            # if df.loc[j,'Target class'] == 3 or df.loc[j,'Target class'] == 4:

                
                # if df.loc[j,video_pred_str] == 3 or df.loc[j,video_pred_str] == 4:
                    # n_vid_corr_act_frames += 1
                
            # elif df.loc[j,'Target class'] == 1 or df.loc[j,'Target class'] == 2:

                
                # if df.loc[j,video_pred_str] == 1 or df.loc[j,video_pred_str] == 2:
                    # n_vid_corr_inact_frames += 1
                
                
                
                
            # if df.loc[j,'Target class'] == 1:

                
                # if df.loc[j,video_pred_str] == 1:
                    # n_vid_corr_negInact_frames += 1
            
            # elif df.loc[j,'Target class'] == 2:

                
                # if df.loc[j,video_pred_str] == 2:
                    # n_vid_corr_posInact_frames += 1
            
            # elif df.loc[j,'Target class'] == 3:

                
                # if df.loc[j,video_pred_str] == 3:
                    # n_vid_corr_posAct_frames += 1
            
            # elif df.loc[j,'Target class'] == 4:

                
                # if df.loc[j,video_pred_str] == 4:
                    # n_vid_corr_negAct_frames += 1
                    
                    
            
            # if df.loc[j,'Target class'] == df.loc[j,video_pred_str]: 
                # n_vid_corr_tot_frames += 1
        
        
        # ### Check target frames
        
        # if df.loc[j,'Target class'] == 2 or df.loc[j,'Target class'] == 3:
            # n_pos_frames += 1
            
        # elif df.loc[j,'Target class'] == 1 or df.loc[j,'Target class'] == 4:
            # n_neg_frames += 1
            
        # if df.loc[j,'Target class'] == 3 or df.loc[j,'Target class'] == 4:
            # n_act_frames += 1
            
        # elif df.loc[j,'Target class'] == 1 or df.loc[j,'Target class'] == 2:
            # n_inact_frames += 1
            
            
            
            
        # if df.loc[j,'Target class'] == 1:
            # n_negInact_frames += 1
        
        # elif df.loc[j,'Target class'] == 2:
            # n_posInact_frames += 1
        
        # elif df.loc[j,'Target class'] == 3:
            # n_posAct_frames += 1
        
        # elif df.loc[j,'Target class'] == 4:
            # n_negAct_frames += 1
            
            
            
        # # Check target frames with face
        # if df.loc[j,'Has face?'] == 1:
            # if df.loc[j,'Target class'] == 2 or df.loc[j,'Target class'] == 3:
                # n_pos_frames_wFace += 1
                
            # elif df.loc[j,'Target class'] == 1 or df.loc[j,'Target class'] == 4:
                # n_neg_frames_wFace += 1
                
            # if df.loc[j,'Target class'] == 3 or df.loc[j,'Target class'] == 4:
                # n_act_frames_wFace += 1
                
            # elif df.loc[j,'Target class'] == 1 or df.loc[j,'Target class'] == 2:
                # n_inact_frames_wFace += 1
                
                
                
                
            # if df.loc[j,'Target class'] == 1:
                # n_negInact_frames_wFace += 1
            
            # elif df.loc[j,'Target class'] == 2:
                # n_posInact_frames_wFace += 1
            
            # elif df.loc[j,'Target class'] == 3:
                # n_posAct_frames_wFace += 1
            
            # elif df.loc[j,'Target class'] == 4:
                # n_negAct_frames_wFace += 1
            
            
        
        
        
        
        ### Check sound frames with face
        if df.loc[j,text_pred_str] != 0:
            if df.loc[j,'Has face?'] == 1:
                n_sound_frames_with_face += 1
                
                
        # ### Neutral frames
        
        # if df.loc[j,'Target class'] == 0:
            # n_neutral_frames += 1
        
        
        
        # if df.loc[j,mismatch_str] == 1:
            # if df.loc[j,aud_pred_str] != df.loc[j,text_pred_str]:
                # n_TextAudMismatch_frames += 1
            # if df.loc[j,aud_pred_str] != df.loc[j,video_pred_str]:
                # n_VidAudMismatch_frames += 1
            # if df.loc[j,video_pred_str] != df.loc[j,text_pred_str]:
                # n_VidTextMismatch_frames += 1
                
                
        ### Check mismatches
        
        if df.loc[j,'Has face?'] == 1:
            if df.loc[j,text_pred_str] != 0:
                
                n_possible_mismatch_frames += 1
                
                if df.loc[j,aud_pred_str] != df.loc[j,text_pred_str]:
                    n_TextAudMismatch_frames += 1
                if df.loc[j,aud_pred_str] != df.loc[j,video_pred_str]:
                    n_VidAudMismatch_frames += 1
                if df.loc[j,video_pred_str] != df.loc[j,text_pred_str]:
                    n_VidTextMismatch_frames += 1
       
        #### For adjacent predictions
        
        if j != 0:
            
            if df.loc[j,video_pred_str] == 1:
                if df.loc[j-1,video_pred_str] == 2:
                    n_vid_cat2_bef_cat1 += 1
                elif df.loc[j-1,video_pred_str] == 3:
                    n_vid_cat3_bef_cat1 += 1
                elif df.loc[j-1,video_pred_str] == 4:
                    n_vid_cat4_bef_cat1 += 1
            
            elif df.loc[j,video_pred_str] == 2:
                if df.loc[j-1,video_pred_str] == 1:
                    n_vid_cat1_bef_cat2 += 1
                elif df.loc[j-1,video_pred_str] == 3:
                    n_vid_cat3_bef_cat2 += 1
                elif df.loc[j-1,video_pred_str] == 4:
                    n_vid_cat4_bef_cat2 += 1
            
            elif df.loc[j,video_pred_str] == 3:
                if df.loc[j-1,video_pred_str] == 2:
                    n_vid_cat2_bef_cat3 += 1
                elif df.loc[j-1,video_pred_str] == 1:
                    n_vid_cat1_bef_cat3 += 1
                elif df.loc[j-1,video_pred_str] == 4:
                    n_vid_cat4_bef_cat3 += 1
            
            elif df.loc[j,video_pred_str] == 4:
                if df.loc[j-1,video_pred_str] == 2:
                    n_vid_cat2_bef_cat4 += 1
                elif df.loc[j-1,video_pred_str] == 3:
                    n_vid_cat3_bef_cat4 += 1
                elif df.loc[j-1,video_pred_str] == 1:
                    n_vid_cat1_bef_cat4 += 1
                    
            
            
            
            
            
            if df.loc[j,aud_pred_str] == 1:
                if df.loc[j-1,aud_pred_str] == 2:
                    n_aud_cat2_bef_cat1 += 1
                elif df.loc[j-1,aud_pred_str] == 3:
                    n_aud_cat3_bef_cat1 += 1
                elif df.loc[j-1,aud_pred_str] == 4:
                    n_aud_cat4_bef_cat1 += 1
            
            elif df.loc[j,aud_pred_str] == 2:
                if df.loc[j-1,aud_pred_str] == 1:
                    n_aud_cat1_bef_cat2 += 1
                elif df.loc[j-1,aud_pred_str] == 3:
                    n_aud_cat3_bef_cat2 += 1
                elif df.loc[j-1,aud_pred_str] == 4:
                    n_aud_cat4_bef_cat2 += 1
            
            elif df.loc[j,aud_pred_str] == 3:
                if df.loc[j-1,aud_pred_str] == 2:
                    n_aud_cat2_bef_cat3 += 1
                elif df.loc[j-1,aud_pred_str] == 1:
                    n_aud_cat1_bef_cat3 += 1
                elif df.loc[j-1,aud_pred_str] == 4:
                    n_aud_cat4_bef_cat3 += 1
            
            elif df.loc[j,aud_pred_str] == 4:
                if df.loc[j-1,aud_pred_str] == 2:
                    n_aud_cat2_bef_cat4 += 1
                elif df.loc[j-1,aud_pred_str] == 3:
                    n_aud_cat3_bef_cat4 += 1
                elif df.loc[j-1,aud_pred_str] == 1:
                    n_aud_cat1_bef_cat4 += 1
                    
            


            
            if df.loc[j,text_pred_str] == 1:
                if df.loc[j-1,text_pred_str] == 2:
                    n_text_cat2_bef_cat1 += 1
                elif df.loc[j-1,text_pred_str] == 3:
                    n_text_cat3_bef_cat1 += 1
                elif df.loc[j-1,text_pred_str] == 4:
                    n_text_cat4_bef_cat1 += 1
            
            elif df.loc[j,text_pred_str] == 2:
                if df.loc[j-1,text_pred_str] == 1:
                    n_text_cat1_bef_cat2 += 1
                elif df.loc[j-1,text_pred_str] == 3:
                    n_text_cat3_bef_cat2 += 1
                elif df.loc[j-1,text_pred_str] == 4:
                    n_text_cat4_bef_cat2 += 1
            
            elif df.loc[j,text_pred_str] == 3:
                if df.loc[j-1,text_pred_str] == 2:
                    n_text_cat2_bef_cat3 += 1
                elif df.loc[j-1,text_pred_str] == 1:
                    n_text_cat1_bef_cat3 += 1
                elif df.loc[j-1,text_pred_str] == 4:
                    n_text_cat4_bef_cat3 += 1
            
            elif df.loc[j,text_pred_str] == 4:
                if df.loc[j-1,text_pred_str] == 2:
                    n_text_cat2_bef_cat4 += 1
                elif df.loc[j-1,text_pred_str] == 3:
                    n_text_cat3_bef_cat4 += 1
                elif df.loc[j-1,text_pred_str] == 1:
                    n_text_cat1_bef_cat4 += 1
                    
            
                    
        
        ### For total score
        
        if df.loc[j,'Has face?'] == 1:
            vid_bool = True
        else:
            vid_bool = False
            
        if df.loc[j,aud_pred_str] == 0:
            aud_bool = False
        else:
            aud_bool = True
            
        if df.loc[j,text_pred_str] == 0:
            text_bool = False
        else:
            text_bool = True
        
        
        
        # score_tot = vid_score_tot * w_vid + aud_score_tot * w_aud + text_score_tot * w_text
        
        
        if vid_bool == False and aud_bool == False and text_bool == False:
            score_tot = score_tot + 0
        
        elif vid_bool == False and aud_bool == False and text_bool == True:
            score_tot = score_tot + this_score_text
            n_scorable_frames += 1
        
        elif vid_bool == False and aud_bool == True and text_bool == False:
            score_tot = score_tot + this_score_aud
            n_scorable_frames += 1

        elif vid_bool == False and aud_bool == True and text_bool == True:
            temp_score = this_score_aud * 0.50 + this_score_text * 0.50
            score_tot = score_tot + temp_score
            n_scorable_frames += 1
        
        elif vid_bool == True and aud_bool == False and text_bool == False:
            score_tot = score_tot + this_score_vid
            n_scorable_frames += 1

        elif vid_bool == True and aud_bool == False and text_bool == True:
            temp_score = this_score_vid * 0.50 + this_score_text * 0.50
            score_tot = score_tot + temp_score
            n_scorable_frames += 1
        
        elif vid_bool == True and aud_bool == True and text_bool == False:
            temp_score = this_score_vid * 0.50 + this_score_aud * 0.50
            score_tot = score_tot + temp_score
            n_scorable_frames += 1
        
        elif vid_bool == True and aud_bool == True and text_bool == True:
            temp_score = this_score_vid * w_vid + this_score_aud * w_aud + this_score_text * w_text
            score_tot = score_tot + temp_score
            n_scorable_frames += 1
        
        
        
    
    # # print(n_neg_frames)
    # # print(str(df.iloc[-1,0]))
    # # print((n_neg_frames/int(df.iloc[-1,0])) * 100)
    
    
    # #n_vid_corr_tot_frames = n_vid_corr_posAct_frames + n_vid_corr_negAct_frames + n_vid_corr_posInact_frames + n_vid_corr_negInact_frames
    
    # #n_aud_corr_tot_frames = n_aud_corr_posAct_frames + n_aud_corr_negAct_frames + n_aud_corr_posInact_frames + n_aud_corr_negInact_frames
    
    # #n_text_corr_tot_frames = n_text_corr_posAct_frames + n_text_corr_negAct_frames + n_text_corr_posInact_frames + n_text_corr_negInact_frames
    
    # n_nonNeutral_frames = n_frames - n_neutral_frames
    
    # out.loc[index, "Positive frames"] = int(n_pos_frames)
    # out.loc[index, "Negative frames"] = int(n_neg_frames)
    
    # out.loc[index, "Active frames"] = int(n_act_frames)
    # out.loc[index, "Inactive frames"] = int(n_inact_frames)
    
    # out.loc[index, "Positive-active frames"] = int(n_posAct_frames)
    # out.loc[index, "Negative-active frames"] = int(n_negAct_frames)
    
    # out.loc[index, "Positive-inactive frames"] = int(n_posInact_frames)
    # out.loc[index, "Negative-inactive frames"] = int(n_negInact_frames)
    
    
    # out.loc[index, "Positive frames with face"] = int(n_pos_frames_wFace)
    # out.loc[index, "Negative frames with face"] = int(n_neg_frames_wFace)
    
    # out.loc[index, "Active frames with face"] = int(n_act_frames_wFace)
    # out.loc[index, "Inactive frames with face"] = int(n_inact_frames_wFace)
    
    # out.loc[index, "Positive-active frames with face"] = int(n_posAct_frames_wFace)
    # out.loc[index, "Negative-active frames with face"] = int(n_negAct_frames_wFace)
    
    # out.loc[index, "Positive-inactive frames with face"] = int(n_posInact_frames_wFace)
    # out.loc[index, "Negative-inactive frames with face"] = int(n_negInact_frames_wFace)
    
    
    
    
    
    out.loc[index, "Positive frames from video"] = int(n_pos_frames_from_vid)
    out.loc[index, "Negative frames from video"] = int(n_neg_frames_from_vid)
    
    out.loc[index, "Active frames from video"] = int(n_act_frames_from_vid)
    out.loc[index, "Inactive frames from video"] = int(n_inact_frames_from_vid)
    
    out.loc[index, "Positive-active frames from video"] = int(n_posAct_frames_from_vid)
    out.loc[index, "Negative-active frames from video"] = int(n_negAct_frames_from_vid)
    
    out.loc[index, "Positive-inactive frames from video"] = int(n_posInact_frames_from_vid)
    out.loc[index, "Negative-inactive frames from video"] = int(n_negInact_frames_from_vid)
    
    # pos-act = 1, pos-inact = 0.5, neg-inact = -0.5, neg-act = -1
    
    
    # vid_score_tot = int(n_posAct_frames_from_vid) * 1 + int(n_posInact_frames_from_vid) * 0.5
        # + int(n_negInact_frames_from_vid) * (-0.5) + int(n_negAct_frames_from_vid) * (-1)
    
    vid_score_tot = (int(n_posAct_frames_from_vid) * s_posAct + int(n_posInact_frames_from_vid) * s_posInact
        + int(n_negInact_frames_from_vid) * s_negInact + int(n_negAct_frames_from_vid) * s_negAct)
    out.loc[index, "Video score (total)"] = int(vid_score_tot)
    
    
    # out.loc[index, "Video Correct Positive frames"] = int(n_vid_corr_pos_frames)
    # out.loc[index, "Video Correct Negative frames"] = int(n_vid_corr_neg_frames)
    # out.loc[index, "Video Correct Active frames"] = int(n_vid_corr_act_frames)
    # out.loc[index, "Video Correct Inactive frames"] = int(n_vid_corr_inact_frames)
    
    # out.loc[index, "Video Correct Positive-active frames"] = int(n_vid_corr_posAct_frames)
    # out.loc[index, "Video Correct Negative-active frames"] = int(n_vid_corr_negAct_frames)
    # out.loc[index, "Video Correct Positive-inactive frames"] = int(n_vid_corr_posInact_frames)
    # out.loc[index, "Video Correct Negative-inactive frames"] = int(n_vid_corr_negInact_frames)
    
    # if n_pos_frames != 0:
        # out.loc[index, "Video positive accuracy"] = float((round((n_vid_corr_pos_frames/n_pos_frames)*100,3)))
    # if n_neg_frames != 0:
        # out.loc[index, "Video negative accuracy"] = float((round((n_vid_corr_neg_frames/n_neg_frames)*100,3)))
    # if n_act_frames != 0:
        # out.loc[index, "Video active accuracy"] = float((round((n_vid_corr_act_frames/n_act_frames)*100,3)))
    # if n_inact_frames != 0:
        # out.loc[index, "Video inactive accuracy"] = float((round((n_vid_corr_inact_frames/n_inact_frames)*100,3)))
        
        
    # if (n_pos_frames + n_neg_frames) != 0:
        # out.loc[index, "Video positive-negative accuracy"] = float((round(((n_vid_corr_pos_frames + n_vid_corr_neg_frames)/(n_pos_frames + n_neg_frames))*100,3)))
    # if (n_act_frames + n_inact_frames) != 0:
        # out.loc[index, "Video active-inactive accuracy"] = float((round(((n_vid_corr_act_frames + n_vid_corr_inact_frames)/(n_act_frames + n_inact_frames))*100,3)))
    
    
    # if n_posAct_frames != 0:
        # out.loc[index, "Video positive-active accuracy"] = float((round((n_vid_corr_posAct_frames/n_posAct_frames)*100,3)))
    # if n_negAct_frames != 0:
        # out.loc[index, "Video negative-active accuracy"] = float((round((n_vid_corr_negAct_frames/n_negAct_frames)*100,3)))
    # if n_posInact_frames != 0:
        # out.loc[index, "Video positive-inactive accuracy"] = float((round((n_vid_corr_posInact_frames/n_posInact_frames)*100,3)))
    # if n_negInact_frames != 0:
        # out.loc[index, "Video negative-inactive accuracy"] = float((round((n_vid_corr_negInact_frames/n_negInact_frames)*100,3)))
        
    # out.loc[index, "Video overall accuracy"] = float((round((n_vid_corr_tot_frames/n_frames_with_face)*100,3)))
    
    out.loc[index, "Video score (%)"] = float((round((vid_score_tot/n_frames_with_face)*100,3)))
    
    # out.loc[index, "Neutral frames"] = int(n_neutral_frames)
    # out.loc[index, "Non-neutral frames"] = int(n_nonNeutral_frames)
    out.loc[index, "Frames with face"] = int(n_frames_with_face)
    
    # out.loc[index, 'Video Correct total frames'] = int(n_vid_corr_tot_frames)
    # out.loc[index, 'Audio Correct total frames' + title_suffix] = int(n_aud_corr_tot_frames)
    # out.loc[index, 'Text Correct total frames' + title_suffix] = int(n_text_corr_tot_frames)
    
    
    #### For audio
    out.loc[index, "Sound frames" + title_suffix] = int(n_sound_frames)
    
    out.loc[index, "Sound frames with face" + title_suffix] = int(n_sound_frames_with_face)
    
    # out.loc[index, "Positive sound frames" + title_suffix] = int(n_pos_sound_frames)
    # out.loc[index, "Negative sound frames" + title_suffix] = int(n_neg_sound_frames)
    
    # out.loc[index, "Active sound frames" + title_suffix] = int(n_act_sound_frames)
    # out.loc[index, "Inactive sound frames" + title_suffix] = int(n_inact_sound_frames)
    
    # out.loc[index, "Positive-active sound frames" + title_suffix] = int(n_posAct_sound_frames)
    # out.loc[index, "Negative-active sound frames" + title_suffix] = int(n_negAct_sound_frames)
    
    # out.loc[index, "Positive-inactive sound frames" + title_suffix] = int(n_posInact_sound_frames)
    # out.loc[index, "Negative-inactive sound frames" + title_suffix] = int(n_negInact_sound_frames)

    out.loc[index, "Positive sound frames from audio" + title_suffix] = int(n_pos_sound_frames_from_aud)
    out.loc[index, "Negative sound frames from audio" + title_suffix] = int(n_neg_sound_frames_from_aud)
    
    out.loc[index, "Active sound frames from audio" + title_suffix] = int(n_act_sound_frames_from_aud)
    out.loc[index, "Inactive sound frames from audio" + title_suffix] = int(n_inact_sound_frames_from_aud)
    
    out.loc[index, "Positive-active sound frames from audio" + title_suffix] = int(n_posAct_sound_frames_from_aud)
    out.loc[index, "Negative-active sound frames from audio" + title_suffix] = int(n_negAct_sound_frames_from_aud)
    
    out.loc[index, "Positive-inactive sound frames from audio" + title_suffix] = int(n_posInact_sound_frames_from_aud)
    out.loc[index, "Negative-inactive sound frames from audio" + title_suffix] = int(n_negInact_sound_frames_from_aud)
    
    aud_score_tot = (int(n_posAct_sound_frames_from_aud) * s_posAct + int(n_posInact_sound_frames_from_aud) * s_posInact
        + int(n_negInact_sound_frames_from_aud) * s_negInact + int(n_negAct_sound_frames_from_aud) * s_negAct)
    out.loc[index, "Audio score (total)" + title_suffix] = int(aud_score_tot)

    # out.loc[index, "Audio Correct Positive frames" + title_suffix] = int(n_aud_corr_pos_frames)
    # out.loc[index, "Audio Correct Negative frames" + title_suffix] = int(n_aud_corr_neg_frames)
    # out.loc[index, "Audio Correct Active frames" + title_suffix] = int(n_aud_corr_act_frames)
    # out.loc[index, "Audio Correct Inactive frames" + title_suffix] = int(n_aud_corr_inact_frames)
    
    # out.loc[index, "Audio Correct Positive-active frames" + title_suffix] = int(n_aud_corr_posAct_frames)
    # out.loc[index, "Audio Correct Negative-active frames" + title_suffix] = int(n_aud_corr_negAct_frames)
    # out.loc[index, "Audio Correct Positive-inactive frames" + title_suffix] = int(n_aud_corr_posInact_frames)
    # out.loc[index, "Audio Correct Negative-inactive frames" + title_suffix] = int(n_aud_corr_negInact_frames)
    
    
    
    out.loc[index, 'Possible mismatch frames' + title_suffix] = int(n_possible_mismatch_frames)
    out.loc[index, "VidAudMismatch" + title_suffix] = int(n_VidAudMismatch_frames)
    out.loc[index, "VidTextMismatch" + title_suffix] = int(n_VidTextMismatch_frames)
    out.loc[index, "TextAudMismatch" + title_suffix] = int(n_TextAudMismatch_frames)
    
    out.loc[index, 'Scorable frames' + title_suffix] = int(n_scorable_frames)
    
    # if n_pos_sound_frames != 0:
        # out.loc[index, "Audio positive accuracy" + title_suffix] = float((round((n_aud_corr_pos_frames/n_pos_sound_frames)*100,3)))
    # if n_neg_sound_frames != 0:
        # out.loc[index, "Audio negative accuracy" + title_suffix] = float((round((n_aud_corr_neg_frames/n_neg_sound_frames)*100,3)))
    # if n_act_sound_frames != 0:
        # out.loc[index, "Audio active accuracy" + title_suffix] = float((round((n_aud_corr_act_frames/n_act_sound_frames)*100,3)))
    # if n_inact_sound_frames != 0:
        # out.loc[index, "Audio inactive accuracy" + title_suffix] = float((round((n_aud_corr_inact_frames/n_inact_sound_frames)*100,3)))
        
        
    # if (n_pos_sound_frames + n_neg_sound_frames) != 0:
        # out.loc[index, "Audio positive-negative accuracy" + title_suffix] = float((round(((n_aud_corr_pos_frames + n_aud_corr_neg_frames)/(n_pos_sound_frames + n_neg_sound_frames))*100,3)))
    # if (n_act_sound_frames + n_inact_sound_frames) != 0:
        # out.loc[index, "Audio active-inactive accuracy" + title_suffix] = float((round(((n_aud_corr_act_frames + n_aud_corr_inact_frames)/(n_act_sound_frames + n_inact_sound_frames))*100,3)))
    
    
    # if n_posAct_sound_frames != 0:
        # out.loc[index, "Audio positive-active accuracy" + title_suffix] = float((round((n_aud_corr_posAct_frames/n_posAct_sound_frames)*100,3)))
    # if n_negAct_sound_frames != 0:
        # out.loc[index, "Audio negative-active accuracy" + title_suffix] = float((round((n_aud_corr_negAct_frames/n_negAct_sound_frames)*100,3)))
    # if n_posInact_sound_frames != 0:
        # out.loc[index, "Audio positive-inactive accuracy" + title_suffix] = float((round((n_aud_corr_posInact_frames/n_posInact_sound_frames)*100,3)))
    # if n_negInact_sound_frames != 0:
        # out.loc[index, "Audio negative-inactive accuracy" + title_suffix] = float((round((n_aud_corr_negInact_frames/n_negInact_sound_frames)*100,3)))
        
    # out.loc[index, "Audio overall accuracy" + title_suffix] = float((round((n_aud_corr_tot_frames/n_sound_frames)*100,3)))
    
    out.loc[index, "Audio score (%)" + title_suffix] = float((round((aud_score_tot/n_sound_frames)*100,3)))
    
    
    
    
    
    
    
    
    #### For text
    #out.loc[index, "Sound frames (2 s)"] = str(n_sound_frames)
    
    out.loc[index, "Positive sound frames from text" + title_suffix] = int(n_pos_sound_frames_from_text)
    out.loc[index, "Negative sound frames from text" + title_suffix] = int(n_neg_sound_frames_from_text)
    
    out.loc[index, "Active sound frames from text" + title_suffix] = int(n_act_sound_frames_from_text)
    out.loc[index, "Inactive sound frames from text" + title_suffix] = int(n_inact_sound_frames_from_text)
    
    out.loc[index, "Positive-active sound frames from text" + title_suffix] = int(n_posAct_sound_frames_from_text)
    out.loc[index, "Negative-active sound frames from text" + title_suffix] = int(n_negAct_sound_frames_from_text)
    
    out.loc[index, "Positive-inactive sound frames from text" + title_suffix] = int(n_posInact_sound_frames_from_text)
    out.loc[index, "Negative-inactive sound frames from text" + title_suffix] = int(n_negInact_sound_frames_from_text)
    
    text_score_tot = (int(n_posAct_sound_frames_from_text) * s_posAct + int(n_posInact_sound_frames_from_text) * s_posInact
        + int(n_negInact_sound_frames_from_text) * s_negInact + int(n_negAct_sound_frames_from_text) * s_negAct)
    out.loc[index, "Text score (total)" + title_suffix] = int(text_score_tot)
    
    # out.loc[index, "Text Correct Positive frames" + title_suffix] = int(n_text_corr_pos_frames)
    # out.loc[index, "Text Correct Negative frames" + title_suffix] = int(n_text_corr_neg_frames)
    # out.loc[index, "Text Correct Active frames" + title_suffix] = int(n_text_corr_act_frames)
    # out.loc[index, "Text Correct Inactive frames" + title_suffix] = int(n_text_corr_inact_frames)
    
    # out.loc[index, "Text Correct Positive-active frames" + title_suffix] = int(n_text_corr_posAct_frames)
    # out.loc[index, "Text Correct Negative-active frames" + title_suffix] = int(n_text_corr_negAct_frames)
    # out.loc[index, "Text Correct Positive-inactive frames" + title_suffix] = int(n_text_corr_posInact_frames)
    # out.loc[index, "Text Correct Negative-inactive frames" + title_suffix] = int(n_text_corr_negInact_frames)
    
    # if n_pos_sound_frames != 0:
        # out.loc[index, "Text positive accuracy" + title_suffix] = float((round((n_text_corr_pos_frames/n_pos_sound_frames)*100,3)))
    # if n_neg_sound_frames != 0:
        # out.loc[index, "Text negative accuracy" + title_suffix] = float((round((n_text_corr_neg_frames/n_neg_sound_frames)*100,3)))
    # if n_act_sound_frames != 0:
        # out.loc[index, "Text active accuracy" + title_suffix] = float((round((n_text_corr_act_frames/n_act_sound_frames)*100,3)))
    # if n_inact_sound_frames != 0:
        # out.loc[index, "Text inactive accuracy" + title_suffix] = float((round((n_text_corr_inact_frames/n_inact_sound_frames)*100,3)))
        
        
    # if (n_pos_sound_frames + n_neg_sound_frames) != 0:
        # out.loc[index, "Text positive-negative accuracy" + title_suffix] = float((round(((n_text_corr_pos_frames + n_text_corr_neg_frames)/(n_pos_sound_frames + n_neg_sound_frames))*100,3)))
    # if (n_act_sound_frames + n_inact_sound_frames) != 0:
        # out.loc[index, "Text active-inactive accuracy" + title_suffix] = float((round(((n_text_corr_act_frames + n_text_corr_inact_frames)/(n_act_sound_frames + n_inact_sound_frames))*100,3)))
    
    
    # if n_posAct_sound_frames != 0:
        # out.loc[index, "Text positive-active accuracy" + title_suffix] = float((round((n_text_corr_posAct_frames/n_posAct_sound_frames)*100,3)))
    # if n_negAct_sound_frames != 0:
        # out.loc[index, "Text negative-active accuracy" + title_suffix] = float((round((n_text_corr_negAct_frames/n_negAct_sound_frames)*100,3)))
    # if n_posInact_sound_frames != 0:
        # out.loc[index, "Text positive-inactive accuracy" + title_suffix] = float((round((n_text_corr_posInact_frames/n_posInact_sound_frames)*100,3)))
    # if n_negInact_sound_frames != 0:
        # out.loc[index, "Text negative-inactive accuracy" + title_suffix] = float((round((n_text_corr_negInact_frames/n_negInact_sound_frames)*100,3)))
        
    # out.loc[index, "Text overall accuracy" + title_suffix] = float((round((n_text_corr_tot_frames/n_sound_frames)*100,3)))
    
    out.loc[index, "Text score (%)" + title_suffix] = float((round((text_score_tot/n_sound_frames)*100,3)))
    
    
    # out.loc[index, 'Possible mismatch frames' + title_suffix] = int(n_possible_mismatch_frames)
    # out.loc[index, "VidAudMismatch" + title_suffix] = int(n_VidAudMismatch_frames)
    # out.loc[index, "VidTextMismatch" + title_suffix] = int(n_VidTextMismatch_frames)
    # out.loc[index, "TextAudMismatch" + title_suffix] = int(n_TextAudMismatch_frames)
    
    
    out.loc[index, "VidTextMismatch %" + title_suffix] = float((round((n_VidTextMismatch_frames/n_possible_mismatch_frames)*100,3)))
    out.loc[index, "TextAudMismatch %" + title_suffix] = float((round((n_TextAudMismatch_frames/n_possible_mismatch_frames)*100,3)))
    out.loc[index, "VidAudMismatch %" + title_suffix] = float((round((n_VidAudMismatch_frames/n_possible_mismatch_frames)*100,3)))
    
    # score_tot = vid_score_tot * w_vid + aud_score_tot * w_aud + text_score_tot * w_text
    
    out.loc[index, "Total score (total)" + title_suffix] = int(score_tot)
    
    out.loc[index, "Total score (%)" + title_suffix] = float((round((score_tot/n_scorable_frames)*100,3)))
    
    
    
    out.loc[index, 'Video Cat 2 before 1'] = int(n_vid_cat2_bef_cat1)
    out.loc[index, 'Video Cat 3 before 1'] = int(n_vid_cat3_bef_cat1)
    out.loc[index, 'Video Cat 4 before 1'] = int(n_vid_cat4_bef_cat1)
    
    out.loc[index, 'Video Cat 1 before 2'] = int(n_vid_cat1_bef_cat2)
    out.loc[index, 'Video Cat 3 before 2'] = int(n_vid_cat3_bef_cat2)
    out.loc[index, 'Video Cat 4 before 2'] = int(n_vid_cat4_bef_cat2)
    
    out.loc[index, 'Video Cat 2 before 3'] = int(n_vid_cat2_bef_cat3)
    out.loc[index, 'Video Cat 1 before 3'] = int(n_vid_cat1_bef_cat3)
    out.loc[index, 'Video Cat 4 before 3'] = int(n_vid_cat4_bef_cat3)
    
    out.loc[index, 'Video Cat 2 before 4'] = int(n_vid_cat2_bef_cat4)
    out.loc[index, 'Video Cat 3 before 4'] = int(n_vid_cat3_bef_cat4)
    out.loc[index, 'Video Cat 1 before 4'] = int(n_vid_cat1_bef_cat4)
    
    
    
    out.loc[index, 'Audio Cat 2 before 1' + title_suffix] = int(n_aud_cat2_bef_cat1)
    out.loc[index, 'Audio Cat 3 before 1' + title_suffix] = int(n_aud_cat3_bef_cat1)
    out.loc[index, 'Audio Cat 4 before 1' + title_suffix] = int(n_aud_cat4_bef_cat1)
    
    out.loc[index, 'Audio Cat 1 before 2' + title_suffix] = int(n_aud_cat1_bef_cat2)
    out.loc[index, 'Audio Cat 3 before 2' + title_suffix] = int(n_aud_cat3_bef_cat2)
    out.loc[index, 'Audio Cat 4 before 2' + title_suffix] = int(n_aud_cat4_bef_cat2)
    
    out.loc[index, 'Audio Cat 2 before 3' + title_suffix] = int(n_aud_cat2_bef_cat3)
    out.loc[index, 'Audio Cat 1 before 3' + title_suffix] = int(n_aud_cat1_bef_cat3)
    out.loc[index, 'Audio Cat 4 before 3' + title_suffix] = int(n_aud_cat4_bef_cat3)
    
    out.loc[index, 'Audio Cat 2 before 4' + title_suffix] = int(n_aud_cat2_bef_cat4)
    out.loc[index, 'Audio Cat 3 before 4' + title_suffix] = int(n_aud_cat3_bef_cat4)
    out.loc[index, 'Audio Cat 1 before 4' + title_suffix] = int(n_aud_cat1_bef_cat4)
    
    
    out.loc[index, 'Text Cat 2 before 1' + title_suffix] = int(n_text_cat2_bef_cat1)
    out.loc[index, 'Text Cat 3 before 1' + title_suffix] = int(n_text_cat3_bef_cat1)
    out.loc[index, 'Text Cat 4 before 1' + title_suffix] = int(n_text_cat4_bef_cat1)
    
    out.loc[index, 'Text Cat 1 before 2' + title_suffix] = int(n_text_cat1_bef_cat2)
    out.loc[index, 'Text Cat 3 before 2' + title_suffix] = int(n_text_cat3_bef_cat2)
    out.loc[index, 'Text Cat 4 before 2' + title_suffix] = int(n_text_cat4_bef_cat2)
    
    out.loc[index, 'Text Cat 2 before 3' + title_suffix] = int(n_text_cat2_bef_cat3)
    out.loc[index, 'Text Cat 1 before 3' + title_suffix] = int(n_text_cat1_bef_cat3)
    out.loc[index, 'Text Cat 4 before 3' + title_suffix] = int(n_text_cat4_bef_cat3)
    
    out.loc[index, 'Text Cat 2 before 4' + title_suffix] = int(n_text_cat2_bef_cat4)
    out.loc[index, 'Text Cat 3 before 4' + title_suffix] = int(n_text_cat3_bef_cat4)
    out.loc[index, 'Text Cat 1 before 4' + title_suffix] = int(n_text_cat1_bef_cat4)
    
    
    
   
    
def calcTotalsValidity(path, validity):
    global out
    global origin_x, origin_y
    global positive_end_x, positive_end_y
    global active_end_x, active_end_y
    global negative_end_x, negative_end_y
    global passive_end_x, passive_end_y
    global empty_vid_x, empty_vid_y
    global empty_aud_x, empty_aud_y
    global empty_tex_x, empty_tex_y
    global neutral_target_start_x, neutral_target_start_y
    global neutral_target_end_x, neutral_target_end_y
    global total_videos
    global w_vid, w_text, w_aud 
    global s_posAct, s_posInact, s_negInact, s_negAct
    print("Calculating results for " + str(validity) + " s validity...")
    
    suffix = ''
    if validity == '2':
        suffix = ''
    else:
        suffix = '_' + validity
    
    title_suffix = ' (' + validity + ' s)'
    
    out.loc[-1, "Video filename"] = "Total"
    
    #for i in range(0,out.shape[0]):
    
    #print(np.sum(out["Total frames"].to_numpy()))
    #out.loc[-1, "Total frames"] = np.sum(out["Total frames"].to_numpy())
    

    out.loc[-1, "Sound frames" + title_suffix] = out['Sound frames' + title_suffix].sum()
    out.loc[-1, "Sound frames with face" + title_suffix] = out['Sound frames with face' + title_suffix].sum()
    
    out.loc[-1, "Total score (total)" + title_suffix] = out["Total score (total)" + title_suffix].sum()
    
    # out.loc[-1, "Positive sound frames" + title_suffix] = out['Positive sound frames' + title_suffix].sum()
    # out.loc[-1, "Negative sound frames" + title_suffix] = out['Negative sound frames' + title_suffix].sum()
    # out.loc[-1, "Active sound frames" + title_suffix] = out['Active sound frames' + title_suffix].sum()
    # out.loc[-1, "Inactive sound frames" + title_suffix] = out['Inactive sound frames' + title_suffix].sum()
    # out.loc[-1, "Positive-active sound frames" + title_suffix] = out['Positive-active sound frames' + title_suffix].sum()
    # out.loc[-1, "Negative-active sound frames" + title_suffix] = out['Negative-active sound frames' + title_suffix].sum()
    # out.loc[-1, "Positive-inactive sound frames" + title_suffix] = out['Positive-inactive sound frames' + title_suffix].sum()
    # out.loc[-1, "Negative-inactive sound frames" + title_suffix] = out['Negative-inactive sound frames' + title_suffix].sum()
    
    
    out.loc[-1, "Positive sound frames from audio" + title_suffix] = out['Positive sound frames from audio' + title_suffix].sum()
    out.loc[-1, "Negative sound frames from audio" + title_suffix] = out['Negative sound frames from audio' + title_suffix].sum()
    out.loc[-1, "Active sound frames from audio" + title_suffix] = out['Active sound frames from audio' + title_suffix].sum()
    out.loc[-1, "Inactive sound frames from audio" + title_suffix] = out['Inactive sound frames from audio' + title_suffix].sum()
    out.loc[-1, "Positive-active sound frames from audio" + title_suffix] = out['Positive-active sound frames from audio' + title_suffix].sum()
    out.loc[-1, "Negative-active sound frames from audio" + title_suffix] = out['Negative-active sound frames from audio' + title_suffix].sum()
    out.loc[-1, "Positive-inactive sound frames from audio" + title_suffix] = out['Positive-inactive sound frames from audio' + title_suffix].sum()
    out.loc[-1, "Negative-inactive sound frames from audio" + title_suffix] = out['Negative-inactive sound frames from audio' + title_suffix].sum()
    
    out.loc[-1, "Audio score (total)" + title_suffix] = out["Audio score (total)" + title_suffix].sum()
    
    out.loc[-1, "Positive sound frames from text" + title_suffix] = out['Positive sound frames from text' + title_suffix].sum()
    out.loc[-1, "Negative sound frames from text" + title_suffix] = out['Negative sound frames from text' + title_suffix].sum()
    out.loc[-1, "Active sound frames from text" + title_suffix] = out['Active sound frames from text' + title_suffix].sum()
    out.loc[-1, "Inactive sound frames from text" + title_suffix] = out['Inactive sound frames from text' + title_suffix].sum()
    out.loc[-1, "Positive-active sound frames from text" + title_suffix] = out['Positive-active sound frames from text' + title_suffix].sum()
    out.loc[-1, "Negative-active sound frames from text" + title_suffix] = out['Negative-active sound frames from text' + title_suffix].sum()
    out.loc[-1, "Positive-inactive sound frames from text" + title_suffix] = out['Positive-inactive sound frames from text' + title_suffix].sum()
    out.loc[-1, "Negative-inactive sound frames from text" + title_suffix] = out['Negative-inactive sound frames from text' + title_suffix].sum()
    
    out.loc[-1, "Text score (total)" + title_suffix] = out["Text score (total)" + title_suffix].sum()
    
    # out.loc[-1, "Audio Correct Positive frames" + title_suffix] = out['Audio Correct Positive frames' + title_suffix].sum()
    # out.loc[-1, "Audio Correct Negative frames" + title_suffix] = out['Audio Correct Negative frames' + title_suffix].sum()
    # out.loc[-1, "Audio Correct Active frames" + title_suffix] = out['Audio Correct Active frames' + title_suffix].sum()
    # out.loc[-1, "Audio Correct Inactive frames" + title_suffix] = out['Audio Correct Inactive frames' + title_suffix].sum()
    # out.loc[-1, "Audio Correct Positive-active frames" + title_suffix] = out['Audio Correct Positive-active frames' + title_suffix].sum()
    # out.loc[-1, "Audio Correct Negative-active frames" + title_suffix] = out['Audio Correct Negative-active frames' + title_suffix].sum()
    # out.loc[-1, "Audio Correct Positive-inactive frames" + title_suffix] = out['Audio Correct Positive-inactive frames' + title_suffix].sum()
    # out.loc[-1, "Audio Correct Negative-inactive frames" + title_suffix] = out['Audio Correct Negative-inactive frames' + title_suffix].sum()
    # out.loc[-1, "Text Correct Positive frames" + title_suffix] = out['Text Correct Positive frames' + title_suffix].sum()
    # out.loc[-1, "Text Correct Negative frames" + title_suffix] = out['Text Correct Negative frames' + title_suffix].sum()
    # out.loc[-1, "Text Correct Active frames" + title_suffix] = out['Text Correct Active frames' + title_suffix].sum()
    # out.loc[-1, "Text Correct Inactive frames" + title_suffix] = out['Text Correct Inactive frames' + title_suffix].sum()
    # out.loc[-1, "Text Correct Positive-active frames" + title_suffix] = out['Text Correct Positive-active frames' + title_suffix].sum()
    # out.loc[-1, "Text Correct Negative-active frames" + title_suffix] = out['Text Correct Negative-active frames' + title_suffix].sum()
    # out.loc[-1, "Text Correct Positive-inactive frames" + title_suffix] = out['Text Correct Positive-inactive frames' + title_suffix].sum()
    # out.loc[-1, "Text Correct Negative-inactive frames" + title_suffix] = out['Text Correct Negative-inactive frames' + title_suffix].sum()
    
    out.loc[-1, "VidAudMismatch" + title_suffix] = out['VidAudMismatch' + title_suffix].sum()
    out.loc[-1, "VidTextMismatch" + title_suffix] = out['VidTextMismatch' + title_suffix].sum()
    out.loc[-1, "TextAudMismatch" + title_suffix] = out['TextAudMismatch' + title_suffix].sum()
    out.loc[-1, 'Possible mismatch frames' + title_suffix] = out['Possible mismatch frames' + title_suffix].sum()

    # out.loc[-1, 'Audio Correct total frames' + title_suffix] = out['Audio Correct total frames' + title_suffix].sum()
    # out.loc[-1, 'Text Correct total frames' + title_suffix] = out['Text Correct total frames' + title_suffix].sum()
    
    out.loc[-1, 'Scorable frames' + title_suffix] = out['Scorable frames' + title_suffix].sum()

    ### For audio
    # if out.loc[-1, "Positive sound frames" + title_suffix] != 0:
        # out.loc[-1, "Audio positive accuracy" + title_suffix] = float((round((out.loc[-1, "Audio Correct Positive frames" + title_suffix]/out.loc[-1, "Positive sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Negative sound frames" + title_suffix] != 0:
        # out.loc[-1, "Audio negative accuracy" + title_suffix] = float((round((out.loc[-1, "Audio Correct Negative frames" + title_suffix]/out.loc[-1, "Negative sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Active sound frames" + title_suffix] != 0:
        # out.loc[-1, "Audio active accuracy" + title_suffix] = float((round((out.loc[-1, "Audio Correct Active frames" + title_suffix]/out.loc[-1, "Active sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Inactive sound frames" + title_suffix] != 0:
        # out.loc[-1, "Audio inactive accuracy" + title_suffix] = float((round((out.loc[-1, "Audio Correct Inactive frames" + title_suffix]/out.loc[-1, "Inactive sound frames" + title_suffix])*100,3)))
        
        
    # if (out.loc[-1, "Positive sound frames" + title_suffix] + out.loc[-1, "Negative sound frames" + title_suffix]) != 0:
        # out.loc[-1, "Audio positive-negative accuracy" + title_suffix] = float((round(((out.loc[-1, "Audio Correct Positive frames" + title_suffix] + out.loc[-1, "Audio Correct Negative frames" + title_suffix])/(out.loc[-1, "Positive sound frames" + title_suffix] + out.loc[-1, "Negative sound frames" + title_suffix]))*100,3)))
    # if (out.loc[-1, "Active sound frames" + title_suffix] + out.loc[-1, "Inactive sound frames" + title_suffix]) != 0:
        # out.loc[-1, "Audio active-inactive accuracy" + title_suffix] = float((round(((out.loc[-1, "Audio Correct Active frames" + title_suffix] + out.loc[-1, "Audio Correct Inactive frames" + title_suffix])/(out.loc[-1, "Active sound frames" + title_suffix] + out.loc[-1, "Inactive sound frames" + title_suffix]))*100,3)))
    
    
    # if out.loc[-1, "Positive-active sound frames" + title_suffix] != 0:
        # out.loc[-1, "Audio positive-active accuracy" + title_suffix] = float((round((out.loc[-1, "Audio Correct Positive-active frames" + title_suffix]/out.loc[-1, "Positive-active sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Negative-active sound frames" + title_suffix] != 0:
        # out.loc[-1, "Audio negative-active accuracy" + title_suffix] = float((round((out.loc[-1, "Audio Correct Negative-active frames" + title_suffix]/out.loc[-1, "Negative-active sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Positive-inactive sound frames" + title_suffix] != 0:
        # out.loc[-1, "Audio positive-inactive accuracy" + title_suffix] = float((round((out.loc[-1, "Audio Correct Positive-inactive frames" + title_suffix]/out.loc[-1, "Positive-inactive sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Negative-inactive sound frames" + title_suffix] != 0:
        # out.loc[-1, "Audio negative-inactive accuracy" + title_suffix] = float((round((out.loc[-1, "Audio Correct Negative-inactive frames" + title_suffix]/out.loc[-1, "Negative-inactive sound frames" + title_suffix])*100,3)))
        
    # out.loc[-1, "Audio overall accuracy" + title_suffix] = float((round((out.loc[-1, 'Audio Correct total frames' + title_suffix]/out.loc[-1, "Sound frames" + title_suffix])*100,3)))
    
    out.loc[-1, "Audio score (%)" + title_suffix] = float((round((out.loc[-1, 'Audio score (total)' + title_suffix]/out.loc[-1, "Sound frames" + title_suffix])*100,3)))
    
    
    
    ### For text
    # if out.loc[-1, "Positive sound frames" + title_suffix] != 0:
        # out.loc[-1, "Text positive accuracy" + title_suffix] = float((round((out.loc[-1, "Text Correct Positive frames" + title_suffix]/out.loc[-1, "Positive sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Negative sound frames" + title_suffix] != 0:
        # out.loc[-1, "Text negative accuracy" + title_suffix] = float((round((out.loc[-1, "Text Correct Negative frames" + title_suffix]/out.loc[-1, "Negative sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Active sound frames" + title_suffix] != 0:
        # out.loc[-1, "Text active accuracy" + title_suffix] = float((round((out.loc[-1, "Text Correct Active frames" + title_suffix]/out.loc[-1, "Active sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Inactive sound frames" + title_suffix] != 0:
        # out.loc[-1, "Text inactive accuracy" + title_suffix] = float((round((out.loc[-1, "Text Correct Inactive frames" + title_suffix]/out.loc[-1, "Inactive sound frames" + title_suffix])*100,3)))
        
        
    # if (out.loc[-1, "Positive sound frames" + title_suffix] + out.loc[-1, "Negative sound frames" + title_suffix]) != 0:
        # out.loc[-1, "Text positive-negative accuracy" + title_suffix] = float((round(((out.loc[-1, "Text Correct Positive frames" + title_suffix] + out.loc[-1, "Text Correct Negative frames" + title_suffix])/(out.loc[-1, "Positive sound frames" + title_suffix] + out.loc[-1, "Negative sound frames" + title_suffix]))*100,3)))
    # if (out.loc[-1, "Active sound frames" + title_suffix] + out.loc[-1, "Inactive sound frames" + title_suffix]) != 0:
        # out.loc[-1, "Text active-inactive accuracy" + title_suffix] = float((round(((out.loc[-1, "Text Correct Active frames" + title_suffix] + out.loc[-1, "Text Correct Inactive frames" + title_suffix])/(out.loc[-1, "Active sound frames" + title_suffix] + out.loc[-1, "Inactive sound frames" + title_suffix]))*100,3)))
    
    
    # if out.loc[-1, "Positive-active sound frames" + title_suffix] != 0:
        # out.loc[-1, "Text positive-active accuracy" + title_suffix] = float((round((out.loc[-1, "Text Correct Positive-active frames" + title_suffix]/out.loc[-1, "Positive-active sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Negative-active sound frames" + title_suffix] != 0:
        # out.loc[-1, "Text negative-active accuracy" + title_suffix] = float((round((out.loc[-1, "Text Correct Negative-active frames" + title_suffix]/out.loc[-1, "Negative-active sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Positive-inactive sound frames" + title_suffix] != 0:
        # out.loc[-1, "Text positive-inactive accuracy" + title_suffix] = float((round((out.loc[-1, "Text Correct Positive-inactive frames" + title_suffix]/out.loc[-1, "Positive-inactive sound frames" + title_suffix])*100,3)))
    # if out.loc[-1, "Negative-inactive sound frames" + title_suffix] != 0:
        # out.loc[-1, "Text negative-inactive accuracy" + title_suffix] = float((round((out.loc[-1, "Text Correct Negative-inactive frames" + title_suffix]/out.loc[-1, "Negative-inactive sound frames" + title_suffix])*100,3)))
        
    # out.loc[-1, "Text overall accuracy" + title_suffix] = float((round((out.loc[-1, 'Text Correct total frames' + title_suffix]/out.loc[-1, "Sound frames" + title_suffix])*100,3)))
    
    out.loc[-1, "Text score (%)" + title_suffix] = float((round((out.loc[-1, 'Text score (total)' + title_suffix]/out.loc[-1, "Sound frames" + title_suffix])*100,3)))
    
    out.loc[-1, "Total score (%)" + title_suffix] = float((round((out.loc[-1, 'Total score (total)' + title_suffix]/out.loc[-1, "Scorable frames" + title_suffix])*100,3)))
    
    #out.loc[-1, "VidTextMismatch %" + title_suffix] = float((round((n_VidTextMismatch_frames/n_sound_frames_with_face)*100,3)))
    
    out.loc[-1, "VidTextMismatch %" + title_suffix] = float((round((out.loc[-1, 'VidTextMismatch' + title_suffix]/out.loc[-1, 'Possible mismatch frames' + title_suffix])*100,3)))
    out.loc[-1, "TextAudMismatch %" + title_suffix] = float((round((out.loc[-1, 'TextAudMismatch' + title_suffix]/out.loc[-1, 'Possible mismatch frames' + title_suffix])*100,3)))
    out.loc[-1, "VidAudMismatch %" + title_suffix] = float((round((out.loc[-1, 'VidAudMismatch' + title_suffix]/out.loc[-1, 'Possible mismatch frames' + title_suffix])*100,3)))
    
    
    
    out.loc[-1, 'Audio Cat 2 before 1' + title_suffix] = out['Audio Cat 2 before 1' + title_suffix].sum()
    out.loc[-1, 'Audio Cat 3 before 1' + title_suffix] = out['Audio Cat 3 before 1' + title_suffix].sum()
    out.loc[-1, 'Audio Cat 4 before 1' + title_suffix] = out['Audio Cat 4 before 1' + title_suffix].sum()
    
    out.loc[-1, 'Audio Cat 1 before 2' + title_suffix] = out['Audio Cat 1 before 2' + title_suffix].sum()
    out.loc[-1, 'Audio Cat 3 before 2' + title_suffix] = out['Audio Cat 3 before 2' + title_suffix].sum()
    out.loc[-1, 'Audio Cat 4 before 2' + title_suffix] = out['Audio Cat 4 before 2' + title_suffix].sum()
    
    out.loc[-1, 'Audio Cat 2 before 3' + title_suffix] = out['Audio Cat 2 before 3' + title_suffix].sum()
    out.loc[-1, 'Audio Cat 1 before 3' + title_suffix] = out['Audio Cat 1 before 3' + title_suffix].sum()
    out.loc[-1, 'Audio Cat 4 before 3' + title_suffix] = out['Audio Cat 4 before 3' + title_suffix].sum()
    
    out.loc[-1, 'Audio Cat 2 before 4' + title_suffix] = out['Audio Cat 2 before 4' + title_suffix].sum()
    out.loc[-1, 'Audio Cat 3 before 4' + title_suffix] = out['Audio Cat 3 before 4' + title_suffix].sum()
    out.loc[-1, 'Audio Cat 1 before 4' + title_suffix] = out['Audio Cat 1 before 4' + title_suffix].sum()
    
    
    out.loc[-1, 'Text Cat 2 before 1' + title_suffix] = out['Text Cat 2 before 1' + title_suffix].sum()
    out.loc[-1, 'Text Cat 3 before 1' + title_suffix] = out['Text Cat 3 before 1' + title_suffix].sum()
    out.loc[-1, 'Text Cat 4 before 1' + title_suffix] = out['Text Cat 4 before 1' + title_suffix].sum()
    
    out.loc[-1, 'Text Cat 1 before 2' + title_suffix] = out['Text Cat 1 before 2' + title_suffix].sum()
    out.loc[-1, 'Text Cat 3 before 2' + title_suffix] = out['Text Cat 3 before 2' + title_suffix].sum()
    out.loc[-1, 'Text Cat 4 before 2' + title_suffix] = out['Text Cat 4 before 2' + title_suffix].sum()
    
    out.loc[-1, 'Text Cat 2 before 3' + title_suffix] = out['Text Cat 2 before 3' + title_suffix].sum()
    out.loc[-1, 'Text Cat 1 before 3' + title_suffix] = out['Text Cat 1 before 3' + title_suffix].sum()
    out.loc[-1, 'Text Cat 4 before 3' + title_suffix] = out['Text Cat 4 before 3' + title_suffix].sum()
    
    out.loc[-1, 'Text Cat 2 before 4' + title_suffix] = out['Text Cat 2 before 4' + title_suffix].sum()
    out.loc[-1, 'Text Cat 3 before 4' + title_suffix] = out['Text Cat 3 before 4' + title_suffix].sum()
    out.loc[-1, 'Text Cat 1 before 4' + title_suffix] = out['Text Cat 1 before 4' + title_suffix].sum()
    
    
def calcTotals(path):
    global out
    global origin_x, origin_y
    global positive_end_x, positive_end_y
    global active_end_x, active_end_y
    global negative_end_x, negative_end_y
    global passive_end_x, passive_end_y
    global empty_vid_x, empty_vid_y
    global empty_aud_x, empty_aud_y
    global empty_tex_x, empty_tex_y
    global neutral_target_start_x, neutral_target_start_y
    global neutral_target_end_x, neutral_target_end_y
    global total_videos
    global w_vid, w_text, w_aud 
    global s_posAct, s_posInact, s_negInact, s_negAct
    print("Calculating total results...")
    
    # suffix = ''
    # if validity == '2':
        # suffix = ''
    # else:
        # suffix = '_' + validity
    
    # title_suffix = ' (' + validity + ' s)'
    
    out.loc[-1, "Video filename"] = "Total"
    
    #for i in range(0,out.shape[0]):
    
    #print(np.sum(out["Total frames"].to_numpy()))
    #out.loc[-1, "Total frames"] = np.sum(out["Total frames"].to_numpy())
    
    out.loc[-1, "Total frames"] = out['Total frames'].sum()
    # out.loc[-1, "Neutral frames"] = out['Neutral frames'].sum()
    # out.loc[-1, "Non-neutral frames"] = out['Non-neutral frames'].sum()
    out.loc[-1, "Frames with face"] = out['Frames with face'].sum()
    
    # out.loc[-1, "Positive frames"] = out['Positive frames'].sum()
    # out.loc[-1, "Negative frames"] = out['Negative frames'].sum()
    # out.loc[-1, "Active frames"] = out['Active frames'].sum()
    # out.loc[-1, "Inactive frames"] = out['Inactive frames'].sum()
    # out.loc[-1, "Positive-active frames"] = out['Positive-active frames'].sum()
    # out.loc[-1, "Negative-active frames"] = out['Negative-active frames'].sum()
    # out.loc[-1, "Positive-inactive frames"] = out['Positive-inactive frames'].sum()
    # out.loc[-1, "Negative-inactive frames"] = out['Negative-inactive frames'].sum()
    
    
    # out.loc[-1, "Positive frames with face"] = out['Positive frames with face'].sum()
    # out.loc[-1, "Negative frames with face"] = out['Negative frames with face'].sum()
    # out.loc[-1, "Active frames with face"] = out['Active frames with face'].sum()
    # out.loc[-1, "Inactive frames with face"] = out['Inactive frames with face'].sum()
    # out.loc[-1, "Positive-active frames with face"] = out['Positive-active frames with face'].sum()
    # out.loc[-1, "Negative-active frames with face"] = out['Negative-active frames with face'].sum()
    # out.loc[-1, "Positive-inactive frames with face"] = out['Positive-inactive frames with face'].sum()
    # out.loc[-1, "Negative-inactive frames with face"] = out['Negative-inactive frames with face'].sum()
    
    
    
    out.loc[-1, "Positive frames from video"] = out['Positive frames from video'].sum()
    out.loc[-1, "Negative frames from video"] = out['Negative frames from video'].sum()
    out.loc[-1, "Active frames from video"] = out['Active frames from video'].sum()
    out.loc[-1, "Inactive frames from video"] = out['Inactive frames from video'].sum()
    out.loc[-1, "Positive-active frames from video"] = out['Positive-active frames from video'].sum()
    out.loc[-1, "Negative-active frames from video"] = out['Negative-active frames from video'].sum()
    out.loc[-1, "Positive-inactive frames from video"] = out['Positive-inactive frames from video'].sum()
    out.loc[-1, "Negative-inactive frames from video"] = out['Negative-inactive frames from video'].sum()
    
    out.loc[-1, "Video score (total)"] = out['Video score (total)'].sum()
    
    
    
    # out.loc[-1, "Video Correct Positive frames"] = out['Video Correct Positive frames'].sum()
    # out.loc[-1, "Video Correct Negative frames"] = out['Video Correct Negative frames'].sum()
    # out.loc[-1, "Video Correct Active frames"] = out['Video Correct Active frames'].sum()
    # out.loc[-1, "Video Correct Inactive frames"] = out['Video Correct Inactive frames'].sum()
    # out.loc[-1, "Video Correct Positive-active frames"] = out['Video Correct Positive-active frames'].sum()
    # out.loc[-1, "Video Correct Negative-active frames"] = out['Video Correct Negative-active frames'].sum()
    # out.loc[-1, "Video Correct Positive-inactive frames"] = out['Video Correct Positive-inactive frames'].sum()
    # out.loc[-1, "Video Correct Negative-inactive frames"] = out['Video Correct Negative-inactive frames'].sum()
    
    
    # out.loc[-1, 'Video Correct total frames'] = out['Video Correct total frames'].sum()

    
    
    
    
    #print(out.shape)
    ### For video
    # if out.loc[-1, "Positive frames"] != 0:
        # out.loc[-1, "Video positive accuracy"] = float((round((out.loc[-1, "Video Correct Positive frames"]/out.loc[-1, "Positive frames"])*100,3)))
    # if out.loc[-1, "Negative frames"] != 0:
        # out.loc[-1, "Video negative accuracy"] = float((round((out.loc[-1, "Video Correct Negative frames"]/out.loc[-1, "Negative frames"])*100,3)))
    # if out.loc[-1, "Active frames"] != 0:
        # out.loc[-1, "Video active accuracy"] = float((round((out.loc[-1, "Video Correct Active frames"]/out.loc[-1, "Active frames"])*100,3)))
    # if out.loc[-1, "Inactive frames"] != 0:
        # out.loc[-1, "Video inactive accuracy"] = float((round((out.loc[-1, "Video Correct Inactive frames"]/out.loc[-1, "Inactive frames"])*100,3)))
        
        
    # if (out.loc[-1, "Positive frames"] + out.loc[-1, "Negative frames"]) != 0:
        # out.loc[-1, "Video positive-negative accuracy"] = float((round(((out.loc[-1, "Video Correct Positive frames"] + out.loc[-1, "Video Correct Negative frames"])/(out.loc[-1, "Positive frames"] + out.loc[-1, "Negative frames"]))*100,3)))
    # if (out.loc[-1, "Active frames"] + out.loc[-1, "Inactive frames"]) != 0:
        # out.loc[-1, "Video active-inactive accuracy"] = float((round(((out.loc[-1, "Video Correct Active frames"] + out.loc[-1, "Video Correct Inactive frames"])/(out.loc[-1, "Active frames"] + out.loc[-1, "Inactive frames"]))*100,3)))
    
    
    # if out.loc[-1, "Positive-active frames"] != 0:
        # out.loc[-1, "Video positive-active accuracy"] = float((round((out.loc[-1, "Video Correct Positive-active frames"]/out.loc[-1, "Positive-active frames"])*100,3)))
    # if out.loc[-1, "Negative-active frames"] != 0:
        # out.loc[-1, "Video negative-active accuracy"] = float((round((out.loc[-1, "Video Correct Negative-active frames"]/out.loc[-1, "Negative-active frames"])*100,3)))
    # if out.loc[-1, "Positive-inactive frames"] != 0:
        # out.loc[-1, "Video positive-inactive accuracy"] = float((round((out.loc[-1, "Video Correct Positive-inactive frames"]/out.loc[-1, "Positive-inactive frames"])*100,3)))
    # if out.loc[-1, "Negative-inactive frames"] != 0:
        # out.loc[-1, "Video negative-inactive accuracy"] = float((round((out.loc[-1, "Video Correct Negative-inactive frames"]/out.loc[-1, "Negative-inactive frames"])*100,3)))
        
    # out.loc[-1, "Video overall accuracy"] = float((round((out.loc[-1, 'Video Correct total frames']/out.loc[-1, "Frames with face"])*100,3)))
    
    out.loc[-1, "Video score (%)"] = float((round((out.loc[-1, 'Video score (total)']/out.loc[-1, "Frames with face"])*100,3)))
    
    out.loc[-1, 'Video Cat 2 before 1'] = out['Video Cat 2 before 1'].sum()
    out.loc[-1, 'Video Cat 3 before 1'] = out['Video Cat 3 before 1'].sum()
    out.loc[-1, 'Video Cat 4 before 1'] = out['Video Cat 4 before 1'].sum()
    
    out.loc[-1, 'Video Cat 1 before 2'] = out['Video Cat 1 before 2'].sum()
    out.loc[-1, 'Video Cat 3 before 2'] = out['Video Cat 3 before 2'].sum()
    out.loc[-1, 'Video Cat 4 before 2'] = out['Video Cat 4 before 2'].sum()
    
    out.loc[-1, 'Video Cat 2 before 3'] = out['Video Cat 2 before 3'].sum()
    out.loc[-1, 'Video Cat 1 before 3'] = out['Video Cat 1 before 3'].sum()
    out.loc[-1, 'Video Cat 4 before 3'] = out['Video Cat 4 before 3'].sum()
    
    out.loc[-1, 'Video Cat 2 before 4'] = out['Video Cat 2 before 4'].sum()
    out.loc[-1, 'Video Cat 3 before 4'] = out['Video Cat 3 before 4'].sum()
    out.loc[-1, 'Video Cat 1 before 4'] = out['Video Cat 1 before 4'].sum()
    
    

w_vid = 0.33
w_text = 0.33
w_aud = 0.33

s_posAct = 1
s_posInact = 0.5
s_negInact = -0.5
s_negAct = -1    
#dataFolder = "videos"
#dataFolder = "videos\\Processed with dead audio model"

#dataFolder = "videos\\Initial processing results"

#dataFolder = "videos\\Initial processing results 02"

#dataFolder = "videos\\Whisper processing results"
dataFolder = "videos\\Whisper processing results\\no validation"
res = []

# Iterate directory
for file in os.listdir(dataFolder):
    # check only text files
    if file.endswith('_processed_all_data.csv'):
        res.append(file)        

total_videos = len(res)
final_column_names = ['Video filename', 'Total frames', 'Frames with face',
'Total score (%) (0 s)', 'Total score (%) (1 s)', 'Total score (%) (2 s)', 'Total score (%) (3 s)', 'Total score (%) (4 s)',
'Total score (%) (5 s)', 'Total score (%) (6 s)', 'Total score (%) (7 s)', 'Total score (%) (8 s)', 'Total score (%) (9 s)',
'Total score (%) (10 s)',
'Video score (%)',
'Audio score (%) (0 s)', 'Audio score (%) (1 s)', 'Audio score (%) (2 s)', 'Audio score (%) (3 s)', 'Audio score (%) (4 s)',
'Audio score (%) (5 s)', 'Audio score (%) (6 s)', 'Audio score (%) (7 s)', 'Audio score (%) (8 s)', 'Audio score (%) (9 s)',
'Audio score (%) (10 s)',
'Text score (%) (0 s)', 'Text score (%) (1 s)', 'Text score (%) (2 s)', 'Text score (%) (3 s)', 'Text score (%) (4 s)',
'Text score (%) (5 s)', 'Text score (%) (6 s)', 'Text score (%) (7 s)', 'Text score (%) (8 s)', 'Text score (%) (9 s)',
'Text score (%) (10 s)',
'VidTextMismatch % (0 s)', 'VidTextMismatch % (1 s)', 'VidTextMismatch % (2 s)', 'VidTextMismatch % (3 s)', 'VidTextMismatch % (4 s)',
'VidTextMismatch % (5 s)', 'VidTextMismatch % (6 s)', 'VidTextMismatch % (7 s)', 'VidTextMismatch % (8 s)', 'VidTextMismatch % (9 s)',
'VidTextMismatch % (10 s)',
'TextAudMismatch % (0 s)', 'TextAudMismatch % (1 s)', 'TextAudMismatch % (2 s)', 'TextAudMismatch % (3 s)', 'TextAudMismatch % (4 s)',
'TextAudMismatch % (5 s)', 'TextAudMismatch % (6 s)', 'TextAudMismatch % (7 s)', 'TextAudMismatch % (8 s)', 'TextAudMismatch % (9 s)',
'TextAudMismatch % (10 s)',
'VidAudMismatch % (0 s)', 'VidAudMismatch % (1 s)', 'VidAudMismatch % (2 s)', 'VidAudMismatch % (3 s)', 'VidAudMismatch % (4 s)',
'VidAudMismatch % (5 s)', 'VidAudMismatch % (6 s)', 'VidAudMismatch % (7 s)', 'VidAudMismatch % (8 s)', 'VidAudMismatch % (9 s)',
'VidAudMismatch % (10 s)',
'Video Cat 2 before 1', 'Video Cat 3 before 1', 'Video Cat 4 before 1',
'Video Cat 1 before 2', 'Video Cat 3 before 2', 'Video Cat 4 before 2',
'Video Cat 2 before 3', 'Video Cat 1 before 3', 'Video Cat 4 before 3',
'Video Cat 2 before 4', 'Video Cat 3 before 4', 'Video Cat 1 before 4',
'Audio Cat 2 before 1 (0 s)', 'Audio Cat 3 before 1 (0 s)', 'Audio Cat 4 before 1 (0 s)',
'Audio Cat 1 before 2 (0 s)', 'Audio Cat 3 before 2 (0 s)', 'Audio Cat 4 before 2 (0 s)',
'Audio Cat 2 before 3 (0 s)', 'Audio Cat 1 before 3 (0 s)', 'Audio Cat 4 before 3 (0 s)',
'Audio Cat 2 before 4 (0 s)', 'Audio Cat 3 before 4 (0 s)', 'Audio Cat 1 before 4 (0 s)', 
'Text Cat 2 before 1 (0 s)', 'Text Cat 3 before 1 (0 s)', 'Text Cat 4 before 1 (0 s)',
'Text Cat 1 before 2 (0 s)', 'Text Cat 3 before 2 (0 s)', 'Text Cat 4 before 2 (0 s)',
'Text Cat 2 before 3 (0 s)', 'Text Cat 1 before 3 (0 s)', 'Text Cat 4 before 3 (0 s)',
'Text Cat 2 before 4 (0 s)', 'Text Cat 3 before 4 (0 s)', 'Text Cat 1 before 4 (0 s)',
'Audio Cat 2 before 1 (1 s)', 'Audio Cat 3 before 1 (1 s)', 'Audio Cat 4 before 1 (1 s)',
'Audio Cat 1 before 2 (1 s)', 'Audio Cat 3 before 2 (1 s)', 'Audio Cat 4 before 2 (1 s)',
'Audio Cat 2 before 3 (1 s)', 'Audio Cat 1 before 3 (1 s)', 'Audio Cat 4 before 3 (1 s)',
'Audio Cat 2 before 4 (1 s)', 'Audio Cat 3 before 4 (1 s)', 'Audio Cat 1 before 4 (1 s)', 
'Text Cat 2 before 1 (1 s)', 'Text Cat 3 before 1 (1 s)', 'Text Cat 4 before 1 (1 s)',
'Text Cat 1 before 2 (1 s)', 'Text Cat 3 before 2 (1 s)', 'Text Cat 4 before 2 (1 s)',
'Text Cat 2 before 3 (1 s)', 'Text Cat 1 before 3 (1 s)', 'Text Cat 4 before 3 (1 s)',
'Text Cat 2 before 4 (1 s)', 'Text Cat 3 before 4 (1 s)', 'Text Cat 1 before 4 (1 s)',   
'Audio Cat 2 before 1 (2 s)', 'Audio Cat 3 before 1 (2 s)', 'Audio Cat 4 before 1 (2 s)',
'Audio Cat 1 before 2 (2 s)', 'Audio Cat 3 before 2 (2 s)', 'Audio Cat 4 before 2 (2 s)',
'Audio Cat 2 before 3 (2 s)', 'Audio Cat 1 before 3 (2 s)', 'Audio Cat 4 before 3 (2 s)',
'Audio Cat 2 before 4 (2 s)', 'Audio Cat 3 before 4 (2 s)', 'Audio Cat 1 before 4 (2 s)', 
'Text Cat 2 before 1 (2 s)', 'Text Cat 3 before 1 (2 s)', 'Text Cat 4 before 1 (2 s)',
'Text Cat 1 before 2 (2 s)', 'Text Cat 3 before 2 (2 s)', 'Text Cat 4 before 2 (2 s)',
'Text Cat 2 before 3 (2 s)', 'Text Cat 1 before 3 (2 s)', 'Text Cat 4 before 3 (2 s)',
'Text Cat 2 before 4 (2 s)', 'Text Cat 3 before 4 (2 s)', 'Text Cat 1 before 4 (2 s)',   
'Audio Cat 2 before 1 (3 s)', 'Audio Cat 3 before 1 (3 s)', 'Audio Cat 4 before 1 (3 s)',
'Audio Cat 1 before 2 (3 s)', 'Audio Cat 3 before 2 (3 s)', 'Audio Cat 4 before 2 (3 s)',
'Audio Cat 2 before 3 (3 s)', 'Audio Cat 1 before 3 (3 s)', 'Audio Cat 4 before 3 (3 s)',
'Audio Cat 2 before 4 (3 s)', 'Audio Cat 3 before 4 (3 s)', 'Audio Cat 1 before 4 (3 s)', 
'Text Cat 2 before 1 (3 s)', 'Text Cat 3 before 1 (3 s)', 'Text Cat 4 before 1 (3 s)',
'Text Cat 1 before 2 (3 s)', 'Text Cat 3 before 2 (3 s)', 'Text Cat 4 before 2 (3 s)',
'Text Cat 2 before 3 (3 s)', 'Text Cat 1 before 3 (3 s)', 'Text Cat 4 before 3 (3 s)',
'Text Cat 2 before 4 (3 s)', 'Text Cat 3 before 4 (3 s)', 'Text Cat 1 before 4 (3 s)',   
'Audio Cat 2 before 1 (4 s)', 'Audio Cat 3 before 1 (4 s)', 'Audio Cat 4 before 1 (4 s)',
'Audio Cat 1 before 2 (4 s)', 'Audio Cat 3 before 2 (4 s)', 'Audio Cat 4 before 2 (4 s)',
'Audio Cat 2 before 3 (4 s)', 'Audio Cat 1 before 3 (4 s)', 'Audio Cat 4 before 3 (4 s)',
'Audio Cat 2 before 4 (4 s)', 'Audio Cat 3 before 4 (4 s)', 'Audio Cat 1 before 4 (4 s)', 
'Text Cat 2 before 1 (4 s)', 'Text Cat 3 before 1 (4 s)', 'Text Cat 4 before 1 (4 s)',
'Text Cat 1 before 2 (4 s)', 'Text Cat 3 before 2 (4 s)', 'Text Cat 4 before 2 (4 s)',
'Text Cat 2 before 3 (4 s)', 'Text Cat 1 before 3 (4 s)', 'Text Cat 4 before 3 (4 s)',
'Text Cat 2 before 4 (4 s)', 'Text Cat 3 before 4 (4 s)', 'Text Cat 1 before 4 (4 s)',   
'Audio Cat 2 before 1 (5 s)', 'Audio Cat 3 before 1 (5 s)', 'Audio Cat 4 before 1 (5 s)',
'Audio Cat 1 before 2 (5 s)', 'Audio Cat 3 before 2 (5 s)', 'Audio Cat 4 before 2 (5 s)',
'Audio Cat 2 before 3 (5 s)', 'Audio Cat 1 before 3 (5 s)', 'Audio Cat 4 before 3 (5 s)',
'Audio Cat 2 before 4 (5 s)', 'Audio Cat 3 before 4 (5 s)', 'Audio Cat 1 before 4 (5 s)', 
'Text Cat 2 before 1 (5 s)', 'Text Cat 3 before 1 (5 s)', 'Text Cat 4 before 1 (5 s)',
'Text Cat 1 before 2 (5 s)', 'Text Cat 3 before 2 (5 s)', 'Text Cat 4 before 2 (5 s)',
'Text Cat 2 before 3 (5 s)', 'Text Cat 1 before 3 (5 s)', 'Text Cat 4 before 3 (5 s)',
'Text Cat 2 before 4 (5 s)', 'Text Cat 3 before 4 (5 s)', 'Text Cat 1 before 4 (5 s)',   
'Audio Cat 2 before 1 (6 s)', 'Audio Cat 3 before 1 (6 s)', 'Audio Cat 4 before 1 (6 s)',
'Audio Cat 1 before 2 (6 s)', 'Audio Cat 3 before 2 (6 s)', 'Audio Cat 4 before 2 (6 s)',
'Audio Cat 2 before 3 (6 s)', 'Audio Cat 1 before 3 (6 s)', 'Audio Cat 4 before 3 (6 s)',
'Audio Cat 2 before 4 (6 s)', 'Audio Cat 3 before 4 (6 s)', 'Audio Cat 1 before 4 (6 s)', 
'Text Cat 2 before 1 (6 s)', 'Text Cat 3 before 1 (6 s)', 'Text Cat 4 before 1 (6 s)',
'Text Cat 1 before 2 (6 s)', 'Text Cat 3 before 2 (6 s)', 'Text Cat 4 before 2 (6 s)',
'Text Cat 2 before 3 (6 s)', 'Text Cat 1 before 3 (6 s)', 'Text Cat 4 before 3 (6 s)',
'Text Cat 2 before 4 (6 s)', 'Text Cat 3 before 4 (6 s)', 'Text Cat 1 before 4 (6 s)',   
'Audio Cat 2 before 1 (7 s)', 'Audio Cat 3 before 1 (7 s)', 'Audio Cat 4 before 1 (7 s)',
'Audio Cat 1 before 2 (7 s)', 'Audio Cat 3 before 2 (7 s)', 'Audio Cat 4 before 2 (7 s)',
'Audio Cat 2 before 3 (7 s)', 'Audio Cat 1 before 3 (7 s)', 'Audio Cat 4 before 3 (7 s)',
'Audio Cat 2 before 4 (7 s)', 'Audio Cat 3 before 4 (7 s)', 'Audio Cat 1 before 4 (7 s)', 
'Text Cat 2 before 1 (7 s)', 'Text Cat 3 before 1 (7 s)', 'Text Cat 4 before 1 (7 s)',
'Text Cat 1 before 2 (7 s)', 'Text Cat 3 before 2 (7 s)', 'Text Cat 4 before 2 (7 s)',
'Text Cat 2 before 3 (7 s)', 'Text Cat 1 before 3 (7 s)', 'Text Cat 4 before 3 (7 s)',
'Text Cat 2 before 4 (7 s)', 'Text Cat 3 before 4 (7 s)', 'Text Cat 1 before 4 (7 s)',   
'Audio Cat 2 before 1 (8 s)', 'Audio Cat 3 before 1 (8 s)', 'Audio Cat 4 before 1 (8 s)',
'Audio Cat 1 before 2 (8 s)', 'Audio Cat 3 before 2 (8 s)', 'Audio Cat 4 before 2 (8 s)',
'Audio Cat 2 before 3 (8 s)', 'Audio Cat 1 before 3 (8 s)', 'Audio Cat 4 before 3 (8 s)',
'Audio Cat 2 before 4 (8 s)', 'Audio Cat 3 before 4 (8 s)', 'Audio Cat 1 before 4 (8 s)', 
'Text Cat 2 before 1 (8 s)', 'Text Cat 3 before 1 (8 s)', 'Text Cat 4 before 1 (8 s)',
'Text Cat 1 before 2 (8 s)', 'Text Cat 3 before 2 (8 s)', 'Text Cat 4 before 2 (8 s)',
'Text Cat 2 before 3 (8 s)', 'Text Cat 1 before 3 (8 s)', 'Text Cat 4 before 3 (8 s)',
'Text Cat 2 before 4 (8 s)', 'Text Cat 3 before 4 (8 s)', 'Text Cat 1 before 4 (8 s)',   
'Audio Cat 2 before 1 (9 s)', 'Audio Cat 3 before 1 (9 s)', 'Audio Cat 4 before 1 (9 s)',
'Audio Cat 1 before 2 (9 s)', 'Audio Cat 3 before 2 (9 s)', 'Audio Cat 4 before 2 (9 s)',
'Audio Cat 2 before 3 (9 s)', 'Audio Cat 1 before 3 (9 s)', 'Audio Cat 4 before 3 (9 s)',
'Audio Cat 2 before 4 (9 s)', 'Audio Cat 3 before 4 (9 s)', 'Audio Cat 1 before 4 (9 s)', 
'Text Cat 2 before 1 (9 s)', 'Text Cat 3 before 1 (9 s)', 'Text Cat 4 before 1 (9 s)',
'Text Cat 1 before 2 (9 s)', 'Text Cat 3 before 2 (9 s)', 'Text Cat 4 before 2 (9 s)',
'Text Cat 2 before 3 (9 s)', 'Text Cat 1 before 3 (9 s)', 'Text Cat 4 before 3 (9 s)',
'Text Cat 2 before 4 (9 s)', 'Text Cat 3 before 4 (9 s)', 'Text Cat 1 before 4 (9 s)',   
'Audio Cat 2 before 1 (10 s)', 'Audio Cat 3 before 1 (10 s)', 'Audio Cat 4 before 1 (10 s)',
'Audio Cat 1 before 2 (10 s)', 'Audio Cat 3 before 2 (10 s)', 'Audio Cat 4 before 2 (10 s)',
'Audio Cat 2 before 3 (10 s)', 'Audio Cat 1 before 3 (10 s)', 'Audio Cat 4 before 3 (10 s)',
'Audio Cat 2 before 4 (10 s)', 'Audio Cat 3 before 4 (10 s)', 'Audio Cat 1 before 4 (10 s)', 
'Text Cat 2 before 1 (10 s)', 'Text Cat 3 before 1 (10 s)', 'Text Cat 4 before 1 (10 s)',
'Text Cat 1 before 2 (10 s)', 'Text Cat 3 before 2 (10 s)', 'Text Cat 4 before 2 (10 s)',
'Text Cat 2 before 3 (10 s)', 'Text Cat 1 before 3 (10 s)', 'Text Cat 4 before 3 (10 s)',
'Text Cat 2 before 4 (10 s)', 'Text Cat 3 before 4 (10 s)', 'Text Cat 1 before 4 (10 s)',   
'Total score (total) (0 s)', 'Total score (total) (1 s)', 'Total score (total) (2 s)', 'Total score (total) (3 s)', 'Total score (total) (4 s)',
'Total score (total) (5 s)', 'Total score (total) (6 s)', 'Total score (total) (7 s)', 'Total score (total) (8 s)', 'Total score (total) (9 s)',
'Total score (total) (10 s)',
'Video score (total)',
'Audio score (total) (0 s)', 'Audio score (total) (1 s)', 'Audio score (total) (2 s)', 'Audio score (total) (3 s)', 'Audio score (total) (4 s)',
'Audio score (total) (5 s)', 'Audio score (total) (6 s)', 'Audio score (total) (7 s)', 'Audio score (total) (8 s)', 'Audio score (total) (9 s)',
'Audio score (total) (10 s)', 
'Text score (total) (0 s)', 'Text score (total) (1 s)', 'Text score (total) (2 s)', 'Text score (total) (3 s)', 'Text score (total) (4 s)',
'Text score (total) (5 s)', 'Text score (total) (6 s)', 'Text score (total) (7 s)', 'Text score (total) (8 s)', 'Text score (total) (9 s)',
'Text score (total) (10 s)', 
'Sound frames (0 s)', 'Sound frames (1 s)', 'Sound frames (2 s)', 'Sound frames (3 s)', 'Sound frames (4 s)',
'Sound frames (5 s)', 'Sound frames (6 s)', 'Sound frames (7 s)', 'Sound frames (8 s)', 'Sound frames (9 s)', 'Sound frames (10 s)',
'Sound frames with face (0 s)', 'Sound frames with face (1 s)', 'Sound frames with face (2 s)', 'Sound frames with face (3 s)', 'Sound frames with face (4 s)',
'Sound frames with face (5 s)', 'Sound frames with face (6 s)', 'Sound frames with face (7 s)', 'Sound frames with face (8 s)', 'Sound frames with face (9 s)', 
'Sound frames with face (10 s)',
'Possible mismatch frames (0 s)', 'Possible mismatch frames (1 s)', 'Possible mismatch frames (2 s)', 'Possible mismatch frames (3 s)',
'Possible mismatch frames (4 s)', 'Possible mismatch frames (5 s)', 'Possible mismatch frames (6 s)', 'Possible mismatch frames (7 s)',
'Possible mismatch frames (8 s)', 'Possible mismatch frames (9 s)', 'Possible mismatch frames (10 s)',
'VidAudMismatch (0 s)', 'VidAudMismatch (1 s)', 'VidAudMismatch (2 s)', 'VidAudMismatch (3 s)', 'VidAudMismatch (4 s)',
'VidAudMismatch (5 s)', 'VidAudMismatch (6 s)', 'VidAudMismatch (7 s)', 'VidAudMismatch (8 s)', 'VidAudMismatch (9 s)',
'VidAudMismatch (10 s)',
'TextAudMismatch (0 s)', 'TextAudMismatch (1 s)', 'TextAudMismatch (2 s)', 'TextAudMismatch (3 s)', 'TextAudMismatch (4 s)',
'TextAudMismatch (5 s)', 'TextAudMismatch (6 s)', 'TextAudMismatch (7 s)', 'TextAudMismatch (8 s)', 'TextAudMismatch (9 s)',
'TextAudMismatch (10 s)',
'VidTextMismatch (0 s)', 'VidTextMismatch (1 s)', 'VidTextMismatch (2 s)', 'VidTextMismatch (3 s)', 'VidTextMismatch (4 s)',
'VidTextMismatch (5 s)', 'VidTextMismatch (6 s)', 'VidTextMismatch (7 s)', 'VidTextMismatch (8 s)', 'VidTextMismatch (9 s)',
'VidTextMismatch (10 s)',
'Scorable frames (0 s)', 'Scorable frames (1 s)', 'Scorable frames (2 s)', 'Scorable frames (3 s)', 'Scorable frames (4 s)',
'Scorable frames (5 s)', 'Scorable frames (6 s)', 'Scorable frames (7 s)', 'Scorable frames (8 s)', 'Scorable frames (9 s)',
'Scorable frames (10 s)',
'Positive frames from video', 'Negative frames from video', 'Active frames from video', 'Inactive frames from video', 
'Positive-active frames from video', 'Negative-active frames from video', 'Positive-inactive frames from video','Negative-inactive frames from video',
'Positive sound frames from audio (0 s)', 'Negative sound frames from audio (0 s)', 'Active sound frames from audio (0 s)', 
'Inactive sound frames from audio (0 s)', 'Positive-active sound frames from audio (0 s)', 'Negative-active sound frames from audio (0 s)', 
'Positive-inactive sound frames from audio (0 s)','Negative-inactive sound frames from audio (0 s)',
'Positive sound frames from audio (1 s)', 'Negative sound frames from audio (1 s)', 'Active sound frames from audio (1 s)', 
'Inactive sound frames from audio (1 s)', 'Positive-active sound frames from audio (1 s)', 'Negative-active sound frames from audio (1 s)', 
'Positive-inactive sound frames from audio (1 s)','Negative-inactive sound frames from audio (1 s)',
'Positive sound frames from audio (2 s)', 'Negative sound frames from audio (2 s)', 'Active sound frames from audio (2 s)', 
'Inactive sound frames from audio (2 s)', 'Positive-active sound frames from audio (2 s)', 'Negative-active sound frames from audio (2 s)', 
'Positive-inactive sound frames from audio (2 s)','Negative-inactive sound frames from audio (2 s)',
'Positive sound frames from audio (3 s)', 'Negative sound frames from audio (3 s)', 'Active sound frames from audio (3 s)', 
'Inactive sound frames from audio (3 s)', 'Positive-active sound frames from audio (3 s)', 'Negative-active sound frames from audio (3 s)', 
'Positive-inactive sound frames from audio (3 s)','Negative-inactive sound frames from audio (3 s)',
'Positive sound frames from audio (4 s)', 'Negative sound frames from audio (4 s)', 'Active sound frames from audio (4 s)', 
'Inactive sound frames from audio (4 s)', 'Positive-active sound frames from audio (4 s)', 'Negative-active sound frames from audio (4 s)', 
'Positive-inactive sound frames from audio (4 s)','Negative-inactive sound frames from audio (4 s)',
'Positive sound frames from audio (5 s)', 'Negative sound frames from audio (5 s)', 'Active sound frames from audio (5 s)', 
'Inactive sound frames from audio (5 s)', 'Positive-active sound frames from audio (5 s)', 'Negative-active sound frames from audio (5 s)', 
'Positive-inactive sound frames from audio (5 s)','Negative-inactive sound frames from audio (5 s)',
'Positive sound frames from audio (6 s)', 'Negative sound frames from audio (6 s)', 'Active sound frames from audio (6 s)', 
'Inactive sound frames from audio (6 s)', 'Positive-active sound frames from audio (6 s)', 'Negative-active sound frames from audio (6 s)', 
'Positive-inactive sound frames from audio (6 s)','Negative-inactive sound frames from audio (6 s)',
'Positive sound frames from audio (7 s)', 'Negative sound frames from audio (7 s)', 'Active sound frames from audio (7 s)', 
'Inactive sound frames from audio (7 s)', 'Positive-active sound frames from audio (7 s)', 'Negative-active sound frames from audio (7 s)', 
'Positive-inactive sound frames from audio (7 s)','Negative-inactive sound frames from audio (7 s)',
'Positive sound frames from audio (8 s)', 'Negative sound frames from audio (8 s)', 'Active sound frames from audio (8 s)', 
'Inactive sound frames from audio (8 s)', 'Positive-active sound frames from audio (8 s)', 'Negative-active sound frames from audio (8 s)', 
'Positive-inactive sound frames from audio (8 s)','Negative-inactive sound frames from audio (8 s)',
'Positive sound frames from audio (9 s)', 'Negative sound frames from audio (9 s)', 'Active sound frames from audio (9 s)', 
'Inactive sound frames from audio (9 s)', 'Positive-active sound frames from audio (9 s)', 'Negative-active sound frames from audio (9 s)', 
'Positive-inactive sound frames from audio (9 s)','Negative-inactive sound frames from audio (9 s)',
'Positive sound frames from audio (10 s)', 'Negative sound frames from audio (10 s)', 'Active sound frames from audio (10 s)', 
'Inactive sound frames from audio (10 s)', 'Positive-active sound frames from audio (10 s)', 'Negative-active sound frames from audio (10 s)', 
'Positive-inactive sound frames from audio (10 s)','Negative-inactive sound frames from audio (10 s)',
'Positive sound frames from text (0 s)', 'Negative sound frames from text (0 s)', 'Active sound frames from text (0 s)', 
'Inactive sound frames from text (0 s)', 'Positive-active sound frames from text (0 s)', 'Negative-active sound frames from text (0 s)', 
'Positive-inactive sound frames from text (0 s)','Negative-inactive sound frames from text (0 s)',
'Positive sound frames from text (1 s)', 'Negative sound frames from text (1 s)', 'Active sound frames from text (1 s)', 
'Inactive sound frames from text (1 s)', 'Positive-active sound frames from text (1 s)', 'Negative-active sound frames from text (1 s)', 
'Positive-inactive sound frames from text (1 s)','Negative-inactive sound frames from text (1 s)',
'Positive sound frames from text (2 s)', 'Negative sound frames from text (2 s)', 'Active sound frames from text (2 s)', 
'Inactive sound frames from text (2 s)', 'Positive-active sound frames from text (2 s)', 'Negative-active sound frames from text (2 s)', 
'Positive-inactive sound frames from text (2 s)','Negative-inactive sound frames from text (2 s)',
'Positive sound frames from text (3 s)', 'Negative sound frames from text (3 s)', 'Active sound frames from text (3 s)', 
'Inactive sound frames from text (3 s)', 'Positive-active sound frames from text (3 s)', 'Negative-active sound frames from text (3 s)', 
'Positive-inactive sound frames from text (3 s)','Negative-inactive sound frames from text (3 s)',
'Positive sound frames from text (4 s)', 'Negative sound frames from text (4 s)', 'Active sound frames from text (4 s)', 
'Inactive sound frames from text (4 s)', 'Positive-active sound frames from text (4 s)', 'Negative-active sound frames from text (4 s)', 
'Positive-inactive sound frames from text (4 s)','Negative-inactive sound frames from text (4 s)',
'Positive sound frames from text (5 s)', 'Negative sound frames from text (5 s)', 'Active sound frames from text (5 s)', 
'Inactive sound frames from text (5 s)', 'Positive-active sound frames from text (5 s)', 'Negative-active sound frames from text (5 s)', 
'Positive-inactive sound frames from text (5 s)','Negative-inactive sound frames from text (5 s)',
'Positive sound frames from text (6 s)', 'Negative sound frames from text (6 s)', 'Active sound frames from text (6 s)', 
'Inactive sound frames from text (6 s)', 'Positive-active sound frames from text (6 s)', 'Negative-active sound frames from text (6 s)', 
'Positive-inactive sound frames from text (6 s)','Negative-inactive sound frames from text (6 s)',
'Positive sound frames from text (7 s)', 'Negative sound frames from text (7 s)', 'Active sound frames from text (7 s)', 
'Inactive sound frames from text (7 s)', 'Positive-active sound frames from text (7 s)', 'Negative-active sound frames from text (7 s)', 
'Positive-inactive sound frames from text (7 s)','Negative-inactive sound frames from text (7 s)',
'Positive sound frames from text (8 s)', 'Negative sound frames from text (8 s)', 'Active sound frames from text (8 s)', 
'Inactive sound frames from text (8 s)', 'Positive-active sound frames from text (8 s)', 'Negative-active sound frames from text (8 s)', 
'Positive-inactive sound frames from text (8 s)','Negative-inactive sound frames from text (8 s)',
'Positive sound frames from text (9 s)', 'Negative sound frames from text (9 s)', 'Active sound frames from text (9 s)', 
'Inactive sound frames from text (9 s)', 'Positive-active sound frames from text (9 s)', 'Negative-active sound frames from text (9 s)', 
'Positive-inactive sound frames from text (9 s)','Negative-inactive sound frames from text (9 s)',
'Positive sound frames from text (10 s)', 'Negative sound frames from text (10 s)', 'Active sound frames from text (10 s)', 
'Inactive sound frames from text (10 s)', 'Positive-active sound frames from text (10 s)', 'Negative-active sound frames from text (10 s)', 
'Positive-inactive sound frames from text (10 s)','Negative-inactive sound frames from text (10 s)'
]
out_np = np.zeros((total_videos+1,len(final_column_names)))
out = pd.DataFrame(data = out_np, columns = final_column_names)

print(str(total_videos) + " processed files found..")
#print(out.shape)


origin_x = 2300
origin_y = 350

positive_end_x = 2600
positive_end_y = 350

active_end_x = 2300
active_end_y = 50

negative_end_x = 2000
negative_end_y = 350

passive_end_x = 2300
passive_end_y = 650

dead_x = origin_x
dead_y = origin_y

empty_vid_x = dead_x
empty_vid_y = dead_y

empty_tex_x = dead_x
empty_tex_y = dead_y

empty_aud_x = dead_x
empty_aud_y = dead_y

# empty_vid_x = 2310
# empty_vid_y = 360

# empty_tex_x = 2300
# empty_tex_y = 350

# empty_aud_x = 2290
# empty_aud_y = 340

neutral_target_start_x = origin_x - 50
neutral_target_start_y = origin_y - 50

neutral_target_end_x = origin_x + 50
neutral_target_end_y = origin_y + 50


# appendInfo(dataFolder, "fall2022_Team03_teamsession2_processed_all_data.csv", 0, '0')
# appendInfo(dataFolder, "fall2022_Team03_teamsession2_processed_all_data.csv", 0, '1')
# appendInfo(dataFolder, "fall2022_Team03_teamsession2_processed_all_data.csv", 0, '2')
# appendInfo(dataFolder, "fall2022_Team03_teamsession2_processed_all_data.csv", 0, '3')
# appendInfo(dataFolder, "fall2022_Team03_teamsession2_processed_all_data.csv", 0, '4')
# appendInfo(dataFolder, "fall2022_Team03_teamsession2_processed_all_data.csv", 0, '5')
# appendInfo(dataFolder, "fall2022_Team03_teamsession2_processed_all_data.csv", 0, '6')
# appendInfo(dataFolder, "fall2022_Team03_teamsession2_processed_all_data.csv", 0, '7')
# appendInfo(dataFolder, "fall2022_Team03_teamsession2_processed_all_data.csv", 0, '8')
# appendInfo(dataFolder, "fall2022_Team03_teamsession2_processed_all_data.csv", 0, '9')
# appendInfo(dataFolder, "fall2022_Team03_teamsession2_processed_all_data.csv", 0, '10')

for i in range(0,len(res)):
    appendInfo(dataFolder, res[i], i, '0')
    appendInfo(dataFolder, res[i], i, '1')
    appendInfo(dataFolder, res[i], i, '2')
    appendInfo(dataFolder, res[i], i, '3')
    appendInfo(dataFolder, res[i], i, '4')
    appendInfo(dataFolder, res[i], i, '5')
    appendInfo(dataFolder, res[i], i, '6')
    appendInfo(dataFolder, res[i], i, '7')
    appendInfo(dataFolder, res[i], i, '8')
    appendInfo(dataFolder, res[i], i, '9')
    appendInfo(dataFolder, res[i], i, '10')

calcTotals(dataFolder)
calcTotalsValidity(dataFolder, '0')
calcTotalsValidity(dataFolder, '1')
calcTotalsValidity(dataFolder, '2')
calcTotalsValidity(dataFolder, '3')
calcTotalsValidity(dataFolder, '4')
calcTotalsValidity(dataFolder, '5')
calcTotalsValidity(dataFolder, '6')
calcTotalsValidity(dataFolder, '7')
calcTotalsValidity(dataFolder, '8')
calcTotalsValidity(dataFolder, '9')
calcTotalsValidity(dataFolder, '10')

#out.iloc[-1, 0] = "Total"
#out.iloc[-1, 1] = str(int(out['Total frames'].sum()))    
# print(res)
# print(len(res))
# print(out)

#print(out.shape)

out.to_excel(dataFolder + "//" + 'Results_RQ3.xlsx', index=False)
