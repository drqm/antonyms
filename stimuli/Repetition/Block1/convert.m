
  list=dir('*.wav');
  num_word=length(list);

     for j = 1:num_word
      
        [y,fs,nbits]=wavread(list(j).name);
   
        x=0;
        i=1;
        while x==0
            a=y(i);
            if a>.01
                number=i;
                x=1;
            else
                i=i+1;
            end
        end
        
        r=y(number:length(y)-number,:);
        r(:,2)=.999;
        i=2;
        while i<length(r)
            r(i,2)=-.999;
            i=i+2;
        end
        
        
        
        
   wavwrite(r,44100,['x' list(j).name]);        
        
     end
      
 
  
 