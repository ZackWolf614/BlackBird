import Blackbird
import sys
import json


functionMap = {
    'newModel' : Blackbird.NewModel,
    'setMCTSConfig' : Blackbird.SetMCTSConfig,
    'getMCTSConfig' : Blackbird.GetMCTSConfig,
    'trainWithNewGames' : Blackbird.TrainWithNewGames
}

if __name__ == '__main__':
    func = functionMap.get(sys.argv[1])
    kwargs = json.loads(sys.argv[2])
    if func is None:
        print({
            'err' : 'Unsupported function: {}'.format(sys.argv[1])
        })
    elif not isinstance(kwargs, dict):
        print({
            'err' : 'Invalid keyword arugments: {}'.format(json.dumps(kwargs))
        })
    else:
        print({
            'data' : func(**kwargs)
        })
        # try:
        # except Exception as e:
        #     print({
        #         'err' : str(e)
        #     })
    print({
        'done' : True
    })

    

