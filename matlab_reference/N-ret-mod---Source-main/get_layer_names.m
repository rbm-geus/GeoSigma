function [layer_names,Nlay,Npreq] = get_layer_names(reg,NPL)

PLnames = cell(1,5);

PLnames{1} = 'Post_Glacial_Ler_Tørv_Gytje_';
PLnames{2} = 'Post_Glacial_Sand_';
PLnames{3} = 'Post_Glacial_Ler_';
PLnames{4} = 'Sen_Glacial_Sand_';
PLnames{5} = 'Sen_Glacial_Ler_';




if strcmp(reg,'AnholtLæsø')
    P_use = [];
    for i = 1:NPL
        P_use = [P_use,cellstr(['Lag',num2str(i),'_Bund_',PLnames{i}(1:end-1)])];
        %keyboard
    end
else
    P_use = [];
    for i = 1:NPL
        P_use = [P_use,PLnames(i)];
    end
end
%    {['Lag1_Bund_',PLnames{1}(1:end-1)]},{['Lag2_Bund_',PLnames{2}(1:end-1)]}...
%            ,{['Lag3_Bund_',PLnames{3}(1:end-1)]},{['Lag4_Bund_',PLnames{4}(1:end-1)]}...
 %           ,{['Lag5_Bund_',PLnames{5}(1:end-1)]}


% end


switch reg
    case 'Jylland'

        layer_names = [{'topo'},P_use,{'0100_Postglacial_toerv_Bund'}...
            {'0200_Kvartaer_sand_Bund'},{'0300_Kvartaer_ler_Bund'},{'0400_Kvartaer_sand_Bund'},{'1100_Kvartaer_ler_Bund'}...
            {'1200_Kvartaer_sand_Bund'},{'1300_Kvartaer_ler_Bund'},{'1400_Kvartaer_sand_Bund'},{'1500_Kvartaer_ler_Bund'}...
            {'2100_Kvartaer_sand_Bund'},{'2200_Kvartaer_ler_Bund'},{'2300_Kvartaer_sand_Bund'},{'2400_Kvartaer_ler_Bund'}...
            {'5100_Maadegruppen_Gram_og_Hodde_Bund'},{'5200_Oevre_Odderup_ODS3_Bund'},{'5300_Oevre_Arnum_ARL3_Bund'},...
            {'5400_Nedre_Odderup_ODS2_Bund'},{'5500_Nedre_Arnum_ARL2_Bund'},{'5600_Bastrup_BADS6_Bund'},{'5700_Klintinghoved_KRL6_Bund'},{'5800_Bastrup_BADS5_Bund'},{'5900_Klintinghoved_KRL5_Bund'},...
            {'6000_Bastrup_BADS4_Bund'},{'6100_Klintinghoved_KRL4_Bund'},{'6200_Bastrup_BADS3_Bund'},{'6300_Klintinghoved_KRL3_Bund'},{'6400_Bastrup_BADS2_Bund'},...
            {'6500_Klintinghoved_KRL2_Bund'},{'6600_Bastrup_BADS1_Bund'},{'6700_Klintinghoved_KRL1_Vejle_Fjord_Bund'},{'6800_Billund_BDS6_BDS9_Bund'},...
            {'6900_Vejle_Fjord_VFL6_Bund'},{'7000_Billund_BDS4_BDS5_Bund'},{'7100_Vejle_Fjord_VFL4_Bund'},{'7200_Billund_BDS3_Bund'},{'7300_Vejle_Fjord_VFL3_Bund'},...
            {'7400_Billund_BDS2_Bund'},{'7500_Vejle_Fjord_VFL2_Bund'},{'7600_Billund_BDS1_Bund'},{'7700_Vejle_Fjord_VFL1_Bund'},{'7800_Billund_BDS0_Bund'},...
            {'8000_Palaeogen_ler_Bund'},{'8500_Danien_Kalk_Bund'},{'9000_Skrivekridt_Bund'},{'9500_Stensalt_Top'}];
            
        Npreq = find(strcmp(layer_names, '5100_Maadegruppen_Gram_og_Hodde_Bund'))-1;

    case 'Fyn'

        layer_names = [{'topo'},P_use,{'ks1t'},{'ks1b'},{'ks2t'},{'ks2b'},{'ks3t'},{'ks3b'},{'preq'},{'kalk'},{'bund'}];
        
        Npreq = find(strcmp(layer_names, 'kalk'))-1;

    case 'Fyn-MST'

        layer_names = [{'DTM_FYN_HAV_100m_resamp_0vedhav0'},{'02_Top_KS1_Adjusted0'},{'03_Bund_KS1_Adjusted0'},{'04_Top_KS2_Adjusted0'},{'05_Bund_KS2_Adjusted0'},{'06_Top_KS3_Adjusted0'},{'07_Bund_KS3_Adjusted0'},{'08_Top_PreQ_Adjusted0'},{'09_Top_Kalk_Adjusted0'},{'bund'}];
        
        Npreq = find(strcmp(layer_names, '09_Top_Kalk_Adjusted0'))-1;


    case 'Sjælland'

        layer_names = [{'topo'},P_use,{'ks1t'},{'ks1b'},{'ks2t'},{'ks2b'},{'ks3t'},{'ks3b'},{'ks4t'},{'ks4b'},{'preq'},{'pl1b'},{'gk1b'},{'dk1b'},{'bund'}];
        Npreq = find(strcmp(layer_names, 'pl1b'))-1;

    case 'AnholtLæsø'
    
        layer_names = [{'dk8_topo100m'},P_use,{'0100_Postglacial_toerv_Bund'}...
            {'0200_Kvartaer_sand_Bund'},{'0300_Kvartaer_ler_Bund'},{'0400_Kvartaer_sand_Bund'},{'1100_Kvartaer_ler_Bund'}...
            {'1200_Kvartaer_sand_Bund'},{'1300_Kvartaer_ler_Bund'},{'1400_Kvartaer_sand_Bund'}];
        Npreq = size(layer_names,2);

end


Nlay = size(layer_names,2);
end