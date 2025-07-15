        fig['layout']['sliders'] = [{
            'pad': {'b': 10},
            'currentvalue': {'visible': False},
            'steps': [
                {
                    'args': [
                        [f'Frame {i}'],
                        {
                            'frame': {'duration': 125, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 75}
                        }
                    ],
                    'label': '',
                    'method': 'animate'
                }
                for i in range(batch_size, total_points + batch_size, batch_size)
            ]
        }]