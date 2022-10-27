# Python program to demonstrate how to
# perform two sample T-test

# Import the library
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

dep = open("dep-selected.txt", "r")
dep_content = dep.read()
dep_content_list = dep_content.split("\n")
dep.close()
dep_floats = [float(x) for x in dep_content_list[:-1]]
# print(dep_floats)
# stats.probplot(dep_floats, dist="norm", plot= plt)
# plt.title("Selected depression symptoms Q-Q Plot")
# plt.savefig("sds.png")
# stats.shapiro(dep_floats)

bert = open("bert.txt", "r")
bert_content = bert.read()
bert_content_list = bert_content.split("\n")
bert.close()
bert_floats = [float(x) for x in bert_content_list[:-1]]
# stats.probplot(bert_floats, dist="norm", plot= plt)
# plt.title("Bert Q-Q Plot")
# plt.savefig("bert.png")
# stats.shapiro(bert_floats)
# print(bert_floats)

# Creating data groups

# bert

group1 = np.array([0.968778801843318, 0.9541858678955454, 0.9494239631336405, 0.9464285714285714, 0.9471106569230472, 0.9699459876543209, 0.947511574074074, 0.946604938271605, 0.9339506172839506, 0.9634645061728394, 0.9487327188940092, 0.9514400921658986, 0.9514976958525346, 0.9594086021505376, 0.9608776970094287, 0.9410300925925926, 0.949826388888889, 0.9577546296296297, 0.9511959876543211, 0.9501157407407408, 0.935042242703533, 0.9516513056835637, 0.9578533026113671, 0.9518817204301077, 0.9494244451728593, 0.9589506172839506, 0.9489197530864197, 0.962596450617284, 0.9605131172839506, 0.9681327160493828, 0.9658218125960061, 0.9555299539170506, 0.9304147465437789, 0.9626728110599078, 0.9543798083412067, 0.9552662037037037, 0.9737268518518517, 0.9417245370370371, 0.9434027777777777, 0.9404128086419752, 0.9525537634408603, 0.9554147465437788, 0.9621543778801843, 0.9665130568356375, 0.949501571447853, 0.9516396604938271, 0.9559220679012346, 0.9427469135802469, 0.9469714506172838, 0.9580246913580246, 0.9586213517665131, 0.9352726574500768, 0.9568548387096774, 0.9686443932411675, 0.9462815494668647, 0.9478202160493827, 0.9499421296296298, 0.956327160493827, 0.9660108024691358, 0.9564621913580247, 0.9528993855606758, 0.9439324116743472, 0.9557795698924731, 0.9488095238095238, 0.9624395040780518, 0.9546874999999999, 0.9542824074074074, 0.9444444444444444, 0.9532407407407407, 0.9603009259259259, 0.9490399385560676, 0.9531490015360983, 0.9533410138248848, 0.9598502304147466, 0.9560573048223203, 0.9429012345679012, 0.9584104938271605, 0.9457368827160495, 0.9601273148148148, 0.9625385802469135, 0.9521505376344086, 0.9750192012288786, 0.9404953917050691, 0.9407642089093702, 0.962805853884272, 0.9425925925925925, 0.9618827160493827, 0.9458140432098765, 0.9570601851851852, 0.9458526234567901, 0.9450844854070661, 0.948521505376344, 0.9394777265745008, 0.9645545314900154, 0.9552281973661377, 0.9550925925925927, 0.9575038580246912, 0.9462962962962963, 0.9521797839506173, 0.962037037037037]
                  )
#
#
group2 = np.array([0.9679147465437788, 0.9535714285714285, 0.9509984639016897, 0.9552035330261137, 0.9491352216416328, 0.9576581790123457, 0.9603780864197531, 0.9462384259259259, 0.9586805555555555, 0.9597993827160494, 0.9606566820276498, 0.9613287250384024, 0.963536866359447, 0.9494815668202764, 0.9658716233152729, 0.9491512345679012, 0.9529706790123458, 0.9551504629629629, 0.9555748456790124, 0.9567901234567902, 0.9515360983102918, 0.9567396313364056, 0.9662250384024578, 0.9550691244239632, 0.9493858820353624, 0.9554976851851852, 0.9537615740740741, 0.9587962962962963, 0.9608024691358025, 0.963792438271605, 0.9685483870967743, 0.9517089093701997, 0.9359447004608294, 0.9623463901689708, 0.9597979291595164, 0.957716049382716, 0.9681327160493827, 0.9511381172839507, 0.9614583333333333, 0.9515046296296296, 0.9552611367127496, 0.962884024577573, 0.9665322580645161, 0.9566436251920123, 0.9594315793532961, 0.9484567901234568, 0.9621913580246914, 0.9479938271604939, 0.9576967592592592, 0.9556905864197532, 0.9484447004608294, 0.9432219662058372, 0.9645929339477727, 0.9679915514592935, 0.9475348514355129, 0.9521412037037037, 0.9603202160493827, 0.9612461419753087, 0.9684992283950618, 0.9526234567901234, 0.9693932411674347, 0.9572772657450076, 0.9538018433179724, 0.9467933947772658, 0.9555752656036096, 0.9581211419753087, 0.9616512345679011, 0.9460841049382717, 0.9584490740740741, 0.9608989197530864, 0.9558179723502305, 0.9533794162826421, 0.9618855606758833, 0.9626728110599078, 0.9543990899099551, 0.934278549382716, 0.9625578703703703, 0.9614969135802469, 0.9561535493827161, 0.9713155864197531, 0.9477534562211981, 0.9748847926267282, 0.9501344086021506, 0.9417818740399385, 0.9689373927462739, 0.9496141975308642, 0.9618827160493827, 0.9611304012345678, 0.9607060185185184, 0.9581983024691358, 0.9520161290322581, 0.9466013824884792, 0.945142089093702, 0.9609831029185868, 0.9654859919403044, 0.9700810185185186, 0.952391975308642, 0.940895061728395, 0.9612268518518519, 0.9632716049382717]
                  )
print(stats.ttest_ind(group2, group1))
# print(np.round(t,4))
# print(np.round(p,4))

# data_group2 = np.array([0.9433881697231212, 0.9149137001078749, 0.9402867673498742, 0.9140546006066734, 0.9419166385799348, 0.9600044938770924, 0.9259878076315196, 0.9358545947166403, 0.943079701964326, 0.931564687288327])

# data_group2 = np.array([15, 17, 14, 17, 14, 8, 12,
#                         19, 19, 14, 17, 22, 24, 16,
#                         13, 16, 13, 18, 15, 13])

# Perform the two sample t-test with equal variances
# print(stats.ttest_ind(a=data_group1, b=data_group2, equal_var=False))