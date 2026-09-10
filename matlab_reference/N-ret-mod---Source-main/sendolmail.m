function sendolmail(to,subject,body,attachments)
%Sends email using MS Outlook. The format of the function is 
%Similar to the SENDMAIL command.
% Create object and set parameters.
h = actxserver('outlook.Application');
mail = h.CreateItem('olMail');
mail.Subject = subject;
mail.To = to;
mail.BodyFormat = 'olFormatHTML';
mail.HTMLBody = body;
% Add attachments, if specified.
if nargin == 4
    for i = 1:length(attachments)
        mail.attachments.Add(attachments{i});
    end
end
% Send message and release object.
mail.Send;
h.release;




% EXAMPLE
% Attachment
% sendolmail('test@mathworks.com','Test link','Test message including a <A HREF=<http://www.mathworks.com>>Link</A>', {'C:\attachment1.txt' 'C:\attachment2.txt'});
% Image
% sendolmail('test@mathworks.com','Test image', 'Test message including an image <img src="http://t3.gstatic.com/images?q=tbn:ANd9GcQ-g_C_RAP7xbdz_Da-GK20YeycTzN2JkZotcIgx22dH2v4cBULmhVdLnc">')