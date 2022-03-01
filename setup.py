from distutils.core import setup

setup(
    name='nlg-eval',
    version='1.1',
    packages=['src', 'src.rouge', 'src.bleurt', 'src.bleurt.lib', 'src.bleurt.wmt', 'src.meteor', 'src.BARTScore',
              'src.questeval', 'src.bert_score', 'src.nubia_score'],
    url='https://github.com/disi-unibo-nlu/nlg-eval',
    license='MIT',
    author='Andrea Zammarchi',
    author_email='andrea.zammarchi3@studio.unibo.it',
    description='NLG evaluation'
)
